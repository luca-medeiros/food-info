import json

DEFAULT_SYSTEM_PROMPT = '''
You are a cook chef/dietitian bot. You are an expert in the food and nutritional domain.
Your mission is provide the best estimations on recipes, and nutritional information all based from the image.
Analyze only the ingredients that can be identified.
Don't include uncertainty, concerns or notes.

Remember that before you answer a question, you must check to see if it complies with your mission
above.
Your mission is to provide information to people about food.
Rules:
- You shouldnt reply to images that dont contain food contents.
'''


def get_prompt(hints=None):
    if hints is not None and hints != '':
        hints = f"Hint: {hints}."
    else:
        hints = ''
    prompt = '''
    Respond in a loadable JSON format.

    Task:
    Reply "imageofFood" false if the image doesn't contain food items.
    Fill the json values with an educated guess if the image is food, what is the menu name, a visual description, and the total nutritional values. 
    The wanted nutrients an their units: calories [kcal], protein [grams], carbohydrates [grams], fat [grams], sodium [milligrams].
    {}

    Response Format:
    '''.format(hints)
    json_format = '''{
    "imageofFood": boolean,
    "menuName": str,
    "description": str,
    "nutrients: [
        {
        "nutrientName": str,
        "nutritionalValue": float,
        "unit": str
        },
    ...]
    }
    * Make sure the JSON is loadable with json.loads()
    '''
    return prompt + json_format


def parse_dictionary_string(input_string):
    input_string = input_string.replace("\n", '')
    try:
        return json.loads(input_string)
    except:
        return {}