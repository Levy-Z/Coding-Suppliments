import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from dataset import *
from custom_llm_for_evaluation import CustomGeminiModel
from LocalRAG import create_whole_pipeline

SYSTEM_PROMPT = """
You are an expert in trading strategies and quantitative finance.
Your role is to answer questions using the retrieved context as the primary source.
If needed, you may use general knowledge, but clearly distinguish it from retrieved evidence.
"""

TEXTBOOKS_FILEPATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books"
VECTORSTORE_SAVE_PATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/"


def docs_to_text_list(retrieved_docs, max_chars=300):
    """
    Convert retrieved Document objects into a list of short text snippets.
    """
    snippets = []
    for doc in retrieved_docs:
        text = doc.page_content.replace("\n", " ").strip()
        snippets.append(text[:max_chars])
    return snippets


def build_rag_chains():
    """
    Build RAG chains for different reasoning modes.
    """
    rag_chain_direct = create_whole_pipeline(
        system_prompt=SYSTEM_PROMPT,
        documents_dir=TEXTBOOKS_FILEPATH,
        vectorstore_save_path=VECTORSTORE_SAVE_PATH,
        reasoning_mode="direct"
    )

    rag_chain_cot = create_whole_pipeline(
        system_prompt=SYSTEM_PROMPT,
        documents_dir=TEXTBOOKS_FILEPATH,
        vectorstore_save_path=VECTORSTORE_SAVE_PATH,
        reasoning_mode="concise_rationale"
    )

    return {
        "direct": rag_chain_direct,
        "cot": rag_chain_cot
    }


def add_actual_outputs_to_dataset(data: pd.DataFrame):
    """
    Run the same inputs through multiple reasoning modes and store outputs.
    """
    rag_chains = build_rag_chains()

    def call_rag_chains(input_prompt):
        direct_response = rag_chains["direct"].invoke({"input": input_prompt})
        cot_response = rag_chains["cot"].invoke({"input": input_prompt})

        retrieved_context = docs_to_text_list(direct_response["retrieved_docs"])

        return pd.Series({
            "direct_output": direct_response["answer"],
            "cot_output": cot_response["answer"],
            "retrieved_context": retrieved_context
        })

    result_cols = data["input"].apply(call_rag_chains)
    data = pd.concat([data, result_cols], axis=1)
    return data


def create_test_cases_from_dataframe(data: pd.DataFrame, output_column: str):
    """
    Convert dataframe rows into DeepEval test cases for one output mode.
    """
    test_cases = []

    for _, row in data.iterrows():
        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row[output_column],
            expected_output=row["expected_output"],
            retrieval_context=row["retrieved_context"],
            context=row["retrieved_context"]
        )
        test_cases.append(test_case)

    return test_cases


def evaluate_reasoning_mode(data: pd.DataFrame, output_column: str, model):
    """
    Evaluate one reasoning mode using DeepEval metrics.
    """
    test_cases = create_test_cases_from_dataframe(data, output_column)

    metrics = [
        FaithfulnessMetric(threshold=0.5, model=model),
        AnswerRelevancyMetric(threshold=0.5, model=model),
        ContextualPrecisionMetric(threshold=0.5, model=model),
        ContextualRecallMetric(threshold=0.5, model=model),
        ContextualRelevancyMetric(threshold=0.5, model=model)
    ]

    dataset = EvaluationDataset(test_cases=test_cases)
    results = dataset.evaluate(metrics=metrics)
    return results


if __name__ == "__main__":
    csv_path = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/eval_dataset.csv"
    no_actual_outputs_df = pd.read_csv(csv_path)

    actual_outputs_df = add_actual_outputs_to_dataset(no_actual_outputs_df)

    print("\n===== SAMPLE COMPARISON =====")
    for i in range(min(3, len(actual_outputs_df))):
        print(f"\nQuestion: {actual_outputs_df.iloc[i]['input']}")
        print(f"\nDirect Output:\n{actual_outputs_df.iloc[i]['direct_output']}")
        print(f"\nCoT Output:\n{actual_outputs_df.iloc[i]['cot_output']}")
        print(f"\nRetrieved Context Preview:\n{actual_outputs_df.iloc[i]['retrieved_context'][:2]}")
        print("-" * 80)

    # Save outputs for manual comparison
    actual_outputs_df.to_csv(
        "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/eval_outputs_comparison.csv",
        index=False
    )

    # Quantitative evaluation
    evaluator_model = CustomGeminiModel()

    print("\n===== EVALUATING DIRECT MODE =====")
    direct_results = evaluate_reasoning_mode(
        actual_outputs_df,
        output_column="direct_output",
        model=evaluator_model
    )

    print("\n===== EVALUATING COT MODE =====")
    cot_results = evaluate_reasoning_mode(
        actual_outputs_df,
        output_column="cot_output",
        model=evaluator_model
    )

    print("\n===== DONE =====")
