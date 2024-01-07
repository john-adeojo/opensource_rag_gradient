import json
import pathlib
from dotenv import load_dotenv
import setup
import os
from gradientai import Gradient

# Set Environment Variables
setup.set_environment_variables()
gradient = Gradient()

model_adapter_id = "f06aec81-7305-435a-bb2f-d7b042da4d8d_model_adapter"
new_model_adapter = gradient.get_model_adapter(model_adapter_id=model_adapter_id)
print(new_model_adapter)

