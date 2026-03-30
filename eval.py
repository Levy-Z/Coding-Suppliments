from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import pandas as pd
from dataset import *
from custom_llm_for_evaluation import CustomGeminiModel
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,  AnswerRelevancyMetric,  FaithfulnessMetric
from LocalRAG import create_whole_pipeline

SYSTEM_PROMPT = ""
TEXTBOOKS_FILEPATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books"

def add_actual_outputs_to_dataset(data : pd.DataFrame):
    rag_chain = create_whole_pipeline(
        system_prompt=SYSTEM_PROMPT,
        documents_dir=TEXTBOOKS_FILEPATH,
        vectorstore_save_path="/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/"
    )

    def call_rag_chain(input_prompt):
        response = rag_chain.invoke({
            "input": input_prompt
        })
        return response['answer'], response['context']

    data[["actual_output", "actual_retreived_context"]] = data["input"].apply(call_rag_chain).apply(pd.Series)
    return data


def create_dataset_from_csv(csv_filepath):
    dataset = EvaluationDataset()

    dataset.add_goldens_from_csv_file(
        file_path=csv_filepath,
        input_col_name="input",
        expected_output_col_name="expected_output",
        retrieval_context_col_name="retreived_context"
    )

    return dataset


if __name__ == "__main__":
    no_actual_outputs_df = pd.read_csv('/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/eval_dataset.csv')
    actual_outputs_df = add_actual_outputs_to_dataset(no_actual_outputs_df)

    print(actual_outputs_df.head())


    # dataset = EvaluationDataset(test_cases=[
    #     LLMTestCase(
    #         input="Woah there",
    #         actual_output="Hi there",
    #         expected_output="Woah there",
    #         context=["Woah there hi there"],
    #         retrieval_context=["Woah there hi there"]
    #     )
    # ])
    # x = CustomGeminiModel()
    # faith = ContextualPrecisionMetric(threshold=0.5, model=x)

    # dataset.evaluate(
    #     metrics=[faith]
    # )