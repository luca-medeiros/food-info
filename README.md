
# Large Language-and-Vision Model for Food Detection and Nutritional Estimation


Combines the power of language and vision processing for food detection and nutritional estimation. With the help of [LLaVa](https://llava-vl.github.io/), it takes an image of food and optional textual hints as input and generates a comprehensive menu name, description, and nutritional information.
The estimations are not realiable and the model halluciates a lot. Just for fun.

<p float="left">
  <img src="/assets/out_pizza.png" width="600" />
  <img src="/assets/out_waffle.png" width="600" /> 
</p>

## Features

- **Language Integration**: By providing optional textual hints about the food image, users can guide the model's predictions, ensuring even more precise results.

- **Menu Generation**: Goes beyond basic food identification. It creates descriptive menu names and descriptions.

- **Nutritional Estimation**: Estimate the nutritional content of the identified foods. This includes calorie count, macronutrient distribution, and other relevant dietary details.

- **User-Friendly Interface**: The user interface is designed to be intuitive and easy to navigate, allowing users to upload images, provide hints, and receive detailed food information effortlessly. Lightning Apps.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- torch (tested 2.0)
- torchvision
- 7875MiB VRAM (4bit)

### Installation
```
git clone https://github.com/luca-medeiros/food-info && cd food-info
pip install -r requirements.txt
```

### Usage

To run the Lightning AI APP:

`lightning run app app.py`


## Acknowledgments

This project is based on the following repositories:

-   [LLaVA](https://llava-vl.github.io/)
-   [Lightning AI](https://github.com/Lightning-AI/lightning)

## [](https://github.com/luca-medeiros/food-info#license)License

This project is licensed under the Apache 2.0 License