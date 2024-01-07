import json
import pathlib
from dotenv import load_dotenv
import setup

# Set Environment Variables
setup.set_environment_variables()
data = "additional_yoda_style_questions_responses_thousand.jsonl"

def read_samples_from_jsonl_in_batches(file_name, batch_size=100):
    # Construct the file path relative to the parent of the current script's directory
    current_dir = pathlib.Path(__file__).parent
    file_path = current_dir.parent / file_name

    batch = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            batch.append(data)
            # Yield the current batch if it has reached the batch size
            if len(batch) == batch_size:
                yield batch
                batch = []

    # Yield the last batch if it has any remaining samples
    if batch:
        yield batch

def example_query_yoda():
    query = """  
        Instruction:\nWhat's the capital of England?\n\n### Response:\n{response}</s> 
    """
    return query

def main():
    load_dotenv()
    from gradientai import Gradient

    sample_query = example_query_yoda()

    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="nous-hermes2-talk-yoda_v2",
        rank= 17,
        learning_rate= 1e-5,
    )
    print(f"Created model adapter with id {new_model_adapter.id}")
    print(f"Asking Sample query: {sample_query}")

    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
    print(f"Generated before finetuning!: {completion}")

    num_epochs = 4
    for epoch in range(num_epochs):
        print(f"Fine-tuning epoch {epoch + 1}")
        batch_number = 0
        for batch in read_samples_from_jsonl_in_batches(data):
            batch_number += 1
            print(f"Processing batch {batch_number} in epoch {epoch + 1}")
            new_model_adapter.fine_tune(samples=batch)

        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=500).generated_output
        print(f"Generated after epoch {epoch + 1}: {completion}")

    # new_model_adapter.delete()
    print("Finetuning complete!")
    gradient.close()

if __name__ == "__main__":
    main()
