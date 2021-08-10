import re
import time
import random
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from mimetypes import guess_type
import cgi
from urllib.parse import parse_qs

import sentencepiece as spm
import torch


MEDIA_FOLDER = Path("media")
FILES_CACHE = {}

WEBSITE_PATTERN = ""

INCORRECT_FORM_ERROR = ""
INGREDIENTS_WITHOUT_DNAME_ERROR = ""
GENERATION_ERROR = ""

DEFAULT_CUISINE_OPTIONS = []


MODEL_FOLDER = Path("model")
TOKENIZER_FILENAME = MODEL_FOLDER / "sp_model.model"
tokenizer = None
cuisines = []

INGREDIENTS_TOKEN = 0
INSTRUCTION_TOKEN = 0

TEXT_LENGTH = 400
DEFAULT_TEMPERATURE = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

MODEL_FILENAME = MODEL_FOLDER / "best_model.pt"
model = None

# CAPITILIZATION EXPRESSION SOURCE:
# https://stackoverflow.com/questions/22800401/how-to-capitalize-the-first-letter-of-every-sentence
CAPITALIZE_PATTERN = re.compile(r'((?<=[.?!]\s)(\w+)|(^\w+))')
ORDERED_LIST_PATTERN = re.compile(r"(\s\d+\.\D)")


# Example recipes names and ingredient lists taken from https://eda.ru/recepty (please, don't sue me...)
EXAMPLE_RECIPES = (
    ("Шоколадный кекс с бананами",
     "Сахар - ¾ стакана, \nКуриное яйцо - 3 штуки, \n"
     "Какао-порошок - 3 столовые ложки, \nПшеничная мука - ¾ стакана, \n"
     "Сливочное масло - 20 г, \nБананы - 3 штуки, \nПанировочные сухари - 30 г"),

    ("Салат с мандаринами и прошутто",
     "Белый бальзамический уксус - 30 мл, \nСоус наршараб - 3 чайные ложки, \n"
     "Оливковое масло - 120 мл, \nМандарины - 4 штуки, \nПрошутто - 100 г, \n"
     "Эстрагон - 15 г, \nСмесь салатных листьев Тоскана «Белая Дача» - 200 г, \n"
     "Сахар - по вкусу, \nСоль - по вкусу, \nМолотый черный перец - по вкусу"),

    ("Овощные котлеты",
     "Свекла - 500 г, \nМорковь - 500 г, \nБелокочанная капуста - 500 г, \nРепчатый лук - 200 г, \n"
     "Манная крупа - 150 г, \nПанировочные сухари - 200 г, \nРастительное масло - 130 мл, \nСоль - по вкусу"),

    ("Салат из капусты с креветками",
     "Чеснок - 5 зубчиков, \nБелокочанная капуста - 200 г, \nКреветки - 70 г, \n"
     "Редис - 20 г, \nКислые яблоки - 1 штука, \nЛук-порей - 2 г, \n"
     "Уксус - 2 столовые ложки, \nОливковое масло - 1 столовая ложка, \n"
     "Сахар - по вкусу, \nРастительное масло - 200 мл, \nПриправа для моркови по-корейски - 2 г, \n"
     "Соль - по вкусу, \nЧерная соль - щепотка, \nМолотый черный перец - по вкусу"),

    ("Шарлотка",
     "Куриное яйцо - 5 штук, \nСахар - 200 г, \nПшеничная мука - 150 г, \n"
     "Разрыхлитель - 1 чайная ложка, \nКислые яблоки - 1 кг"),

    ("Крабово-сырный салат шариками",
     "Сыр - 100 г, \nКрабовые палочки - 200 г, \nКуриное яйцо - 2 штуки, \nЧеснок - 1 зубчик, \nМайонез - по вкусу"),

    ("Кекс с вишней и орехами",
     "Сливочное масло - 200 г, \nСахар - 150 г, \nКуриное яйцо - 4 штуки, \n"
     "Пшеничная мука - 200 г, \nТертая цедра апельсина - 1 чайная ложка, \n"
     "Шоколадный ликер - 1 столовая ложка, \nРазрыхлитель - 1 чайная ложка, \n"
     "Молотая корица - 1 чайная ложка, \nВишня - 300 г, \n"
     "Грецкие орехи - 200 г, \nСахарная пудра - 3 столовые ложки"),

    ("Эскалопы из свинины под сырно-чесночной корочкой",
     "Свинина - 800 г, \nТвердый сыр - 80 г, \nПанировочные сухари - 70 г, \nЯичный желток - 1 штука, \n"
     "Чеснок - 2 зубчика, \nРастительное масло - 3 столовые ложки, \nПетрушка - по вкусу, \n"
     "Смесь салатных листьев - по вкусу, \nСоль - по вкусу, \nМолотый черный перец - по вкусу")
)


def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = ('', 80)
    httpd = server_class(server_address, handler_class)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


def capitalize_re(x):
    # `re.sub` repl for capitalizing the string
    return x.group().capitalize()


def newline_re(x):
    # `re.sub` repl for putting instruction steps on different lines.
    return f"<br>{x.group()}"


def default_select_predicate(index: int, cuisine_name: str):
    # Pick "Русская кухня" as default selected option
    return cuisine_name.startswith("Русская")


def cuisine_id_predicate(id: int):
    # Make a predicate that picks just but id
    def func(index: int, name: str):
        return index == id
    return func


def generate_cuisine_options(select_predicate=default_select_predicate):
    # Fill up the selection box with selected option picked with the select_predicate.
    cuisine_options = "\n".join(f"<option "
                                f"value=\"{i}\" "
                                f"{'selected' if select_predicate(i, name) else ''}>"
                                f"  {name}"
                                f"</option>"
                                for i, name in enumerate(cuisines, 4))
    return cuisine_options


class HttpRoboChef(BaseHTTPRequestHandler):
    def do_GET(self):
        # Process GET requests.

        if self.path == "/":
            # Main page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            random_dname, random_ingredients = random.choice(EXAMPLE_RECIPES)
            self.wfile.write(
                WEBSITE_PATTERN.format(
                    cuisine_options=DEFAULT_CUISINE_OPTIONS,
                    dname=random_dname,
                    ingredients=random_ingredients,
                    temperature=DEFAULT_TEMPERATURE,
                    response=''
                ).encode()
            )
        else:
            # Acquiring resources referenced on the main page (style.css, robochef.svg, script.js)
            path = MEDIA_FOLDER / Path(self.path[1:])
            if path.exists():
                self.send_response(200)
                self.send_header("Content-type", guess_type(path)[0])
                self.end_headers()
                if path in FILES_CACHE:
                    # Send cached resource or load from the disk to update the cache if modification time is more recent
                    data, mtime = FILES_CACHE[path]
                    if path.stat().st_mtime_ns > mtime:
                        data = open(path, "rb").read()
                        FILES_CACHE[path] = (data, path.stat().st_mtime_ns)
                else:
                    # Resource not found in the cache
                    data = open(path, "rb").read()
                    FILES_CACHE[path] = (data, path.stat().st_mtime_ns)
                self.wfile.write(data)
            else:
                # Resource is not found
                self.send_response(404)
                self.end_headers()

    def parse_post(self):
        # Parse the post arguments and return as python dict.
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        postvars = {}
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            symbols = self.rfile.read(length).decode("ascii")
            postvars = parse_qs(symbols, keep_blank_values=True, encoding='utf-8')
        return postvars

    def do_POST(self):
        if self.path != "/":
            # POST requests only allowed to the main page
            self.send_response(403)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        parsed = self.parse_post()
        try:
            # Unfold the expected vars from the post form

            orig_dname = parsed['dname'][0].strip()
            orig_ingredients = parsed['ingredients'][0].strip()

            cuisine = int(parsed['cuisine'][0])
            dname = orig_dname.lower()
            ingredients = orig_ingredients.lower()
            temperature = max(float(parsed['temperature'][0]), 0.01)
        except (KeyError, ValueError, IndexError):
            # Received incorrect data - Response with the incorrect form error page
            random_dname, random_ingredients = random.choice(EXAMPLE_RECIPES)
            self.wfile.write(
                WEBSITE_PATTERN.format(
                    cuisine_options=DEFAULT_CUISINE_OPTIONS,
                    dname=random_dname,
                    ingredients=random_ingredients,
                    temperature=DEFAULT_TEMPERATURE,
                    response=INCORRECT_FORM_ERROR
                ).encode()
            )
            return

        # Cannot generate with empty dname and filled ingredients
        if ingredients and not dname:
            # ... - Response with the error page explaining that dname cannot be empty with ingredients field filled
            self.wfile.write(
                WEBSITE_PATTERN.format(
                    cuisine_options=generate_cuisine_options(cuisine_id_predicate(cuisine)),
                    dname=orig_dname,
                    ingredients=orig_ingredients,
                    temperature=temperature,
                    response=INGREDIENTS_WITHOUT_DNAME_ERROR
                ).encode()
            )
            return

        input_seq: list = [tokenizer.bos_id(), cuisine]
        if dname:
            input_seq.extend(tokenizer.EncodeAsIds(dname))
        if ingredients:
            input_seq.append(INGREDIENTS_TOKEN)
            input_seq.extend(tokenizer.EncodeAsIds(ingredients))

        input_seq: torch.Tensor = torch.tensor([input_seq], dtype=torch.long, device=device)

        start = time.time()
        # Recipe generation loop
        with torch.no_grad():
            while input_seq.size(1) < TEXT_LENGTH and input_seq[0, -1] != tokenizer.eos_id():
                output = model(input_seq, None, False)
                word_weights = output[0, -1].squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.tensor([[word_idx]], dtype=torch.long, device=device)
                input_seq: torch.Tensor = torch.cat((input_seq, word_tensor), 1)
        end = time.time()

        input_seq: list = input_seq[0].tolist()
        print("Generated new text:", end - start, "seconds;", len(input_seq), "words")

        try:
            ingredients_index = input_seq.index(INGREDIENTS_TOKEN)
            instruction_index = input_seq.index(INSTRUCTION_TOKEN, INGREDIENTS_TOKEN + 1)
        except ValueError:
            # Generated text has incorrect format - Response with the generation error page
            self.wfile.write(
                WEBSITE_PATTERN.format(
                    cuisine_options=generate_cuisine_options(cuisine_id_predicate(cuisine)),
                    dname=orig_dname,
                    ingredients=orig_ingredients,
                    temperature=temperature,
                    response=GENERATION_ERROR
                ).encode()
            )
            return

        decoded_dname = tokenizer.Decode(input_seq[2:ingredients_index]).capitalize()               # Dish name
        decoded_ingredients = tokenizer.Decode(input_seq[ingredients_index+1:instruction_index])    # Ingredients
        decoded_steps = tokenizer.Decode(input_seq[instruction_index+1:])                           # Instruction

        # Newlines the ingredient list
        decoded_ingredients = ", <br>".join(line.capitalize() for line in decoded_ingredients.split(", "))

        # Capitalize and put newlines in the instruction
        decoded_steps = CAPITALIZE_PATTERN.sub(capitalize_re, decoded_steps)
        decoded_steps = ORDERED_LIST_PATTERN.sub(newline_re, decoded_steps)

        # Fancy presentation of the result
        result = f'<h2 class="blue-text">{decoded_dname}</h2>' \
                 f'<h5>{tokenizer.IdToPiece(cuisine)}</h5>' \
                 f'<h4>Ингредиенты:</h4>' \
                 f'<p class="flow-text">{decoded_ingredients}</p>' \
                 f'<h4>Инструкция приготовления:</h4>' \
                 f'<p class="flow-text">{decoded_steps}</p>'

        self.wfile.write(
            WEBSITE_PATTERN.format(
                cuisine_options=generate_cuisine_options(cuisine_id_predicate(cuisine)),
                dname=orig_dname,
                ingredients=orig_ingredients,
                temperature=temperature,
                response=f'<h1 class="header">Результат:</h1>{result}'
            ).encode()
        )


if __name__ == "__main__":
    """ NEURAL NETWORK PREPARATION """
    model = torch.load(MODEL_FILENAME, map_location=device)
    model.eval()

    """ SENTENCE TOKENIZER PREPARATION """
    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_FILENAME))
    INGREDIENTS_TOKEN = tokenizer.PieceToId("ИНГРЕДИЕНТЫ:")
    INSTRUCTION_TOKEN = tokenizer.PieceToId("ИНСТРУКЦИЯ ПРИГОТОВЛЕНИЯ:")
    cuisines = [tokenizer.IdToPiece(i) for i in range(4, INGREDIENTS_TOKEN)]

    """ FILLING OUT CONSTANTS """
    DEFAULT_CUISINE_OPTIONS = generate_cuisine_options()
    WEBSITE_PATTERN = open('index.html', "r", encoding="utf-8").read()

    INCORRECT_FORM_ERROR = '<h1 class="header">Ошибка.</h1>' \
                           '<p>Произошла ошибка при считывании заданных в форме значений. ' \
                           'Возможно, выполнен некорректный способ ввода (через сторонние инструменты).</p>'

    INGREDIENTS_WITHOUT_DNAME_ERROR = '<h1 class="header">Ошибка.</h1>' \
                                      '<p>Для использования списка ингредиентов ' \
                                      'нужно обязательно указать название блюда</p>'

    GENERATION_ERROR = '<h1 class="header">Ошибка.</h1><p>Произошла ошибка при генерации рецепта.</p>'

    print("Starting the server!")
    run(handler_class=HttpRoboChef)
