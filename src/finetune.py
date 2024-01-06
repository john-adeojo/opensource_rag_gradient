from dotenv import load_dotenv
import setup
import json 
import pathlib

# Set Environment Variables
setup.set_environment_variables()

def read_samples_from_jsonl(file_name):
    # Construct the file path relative to the parent of the current script's directory
    current_dir = pathlib.Path(__file__).parent
    file_path = current_dir.parent / file_name

    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            samples.append(data)
    return samples

# samples = read_samples_from_jsonl("reformatted_train_v5_reduced_20.jsonl")
# print(samples)

load_dotenv()
from gradientai import Gradient

# def example_query():
#     query = """
#         <s>[INST] <<SYS>>\n context information is below \n--------------------\n['>The trouble is that by removing oil from the face, 
#         the body is forced to spend more resources to produce it again.', 'I\u2019m not confusing anything. The trouble is that by removing oil from the face,
#         the body is forced to spend more resources to produce it again. It is, therefore, of lesser quality, 
#         as it has to substitute essential linoleic acid with oleic acid, which tends to oxidize. Nobody is talking about the exact \u201coverproduction\u201d,
#         as there is no such thing, like hair and nails don\u2019t grow \u201cstronger\u201d as you cut them.',
#         'So it isn\u2019t that more oil is produced \u2013 there\u2019s the same amount of oil. It\u2019s
#         that the skin is damaged and the oil ends up in the wrong place.', 'This is in direct
#         contradiction to your proposal that removing oil from the skin doesn\u2019t cause more oil production.
#         Of course it does, because you broke the surface tension, which is the only stopping mechanism.
#         The key here is to induce a production of a more viscous oil,
#         by lowering triglycerides and increasing unsaturated fatty acid content, primarily linoleic acid.', 
#         \"Your face oil is not hydrating your skin. Never has, never will. 
#         But that doesn't mean you should stop using it.\"]\n--------------------\n\n<</SYS>>\n\n
#         Why doesn't the body produce oil like the face does? [/INST] {Response} </s>
# """
#     return query

def example_query_yoda():
    query = """  
        Instruction:\nWhat's the capital of England?\n\n### Response:\n{response}</s> 
    """
    return query

def main():
    samples = read_samples_from_jsonl("yoda_style_factual_qa_100.jsonl")
    sample_query = example_query_yoda()

    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="nous-hermes2-yoda-v9",
        rank= 16,
        learning_rate= 1e-5,
    )
    print(f"Created model adapter with id {new_model_adapter.id}")
    print(f"Asking Sample query: {sample_query}")

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
    print(f"Generated before finetuning!: {completion}")

    mum_epochs = 25
    count = 0   
    while count < mum_epochs:
        print (f"Fine-tuning epoch {count + 1}")
        new_model_adapter.fine_tune(samples=samples)
        count += 1

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
    print(f"Generated after finetuning!: {completion}")

    # new_model_adapter.delete()
    print("finetuning complete!")
    gradient.close()

if __name__ == "__main__":
    main()