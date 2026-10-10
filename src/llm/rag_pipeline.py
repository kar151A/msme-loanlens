import joblib
import faiss
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# =========================================================
# CONFIGURATION
# =========================================================

ML_MODEL_PATH = "models/loan_model.pkl"

FAISS_INDEX_PATH = (
    "models/msme_loan_faiss.index"
)

METADATA_PATH = (
    "models/faiss_metadata.csv"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading LoanLens models...")


loan_model = joblib.load(
    ML_MODEL_PATH
)


faiss_index = faiss.read_index(
    FAISS_INDEX_PATH
)


metadata = pd.read_csv(
    METADATA_PATH
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


print("Loading language model...")


tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME
)


llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


print("Models loaded successfully.")


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_application(application):

    app = application.copy()

    app[
        "debt_to_revenue_ratio"
    ] = round(
        app["existing_debt_lakhs"]
        / app["annual_revenue_lakhs"],
        2
    )

    app[
        "loan_to_revenue_ratio"
    ] = round(
        app["loan_amount_lakhs"]
        / app["annual_revenue_lakhs"],
        2
    )

    return app


# =========================================================
# ML PREDICTION
# =========================================================

def predict_application(application):

    prepared = prepare_application(
        application
    )

    df = pd.DataFrame(
        [prepared]
    )

    prediction = loan_model.predict(
        df
    )[0]


    probabilities = (
        loan_model.predict_proba(df)[0]
    )


    classes = list(
        loan_model.classes_
    )


    approval_index = classes.index(
        "approved"
    )


    approval_probability = (
        probabilities[
            approval_index
        ]
    )


    return (
        prediction,
        approval_probability
    )


# =========================================================
# APPLICATION TO TEXT
# =========================================================

def application_to_text(application):

    app = prepare_application(
        application
    )


    return (
        f"Business vintage "
        f"{app['business_vintage_months']} months. "

        f"Annual revenue "
        f"{app['annual_revenue_lakhs']} lakh rupees. "

        f"Existing debt "
        f"{app['existing_debt_lakhs']} lakh rupees. "

        f"CIBIL score "
        f"{app['cibil_score']}. "

        f"Sector "
        f"{app['sector']}. "

        f"Requested loan "
        f"{app['loan_amount_lakhs']} lakh rupees. "

        f"Debt to revenue ratio "
        f"{app['debt_to_revenue_ratio']}. "

        f"Loan to revenue ratio "
        f"{app['loan_to_revenue_ratio']}."
    )


# =========================================================
# FAISS RETRIEVAL
# =========================================================

def retrieve_similar_cases(
    application,
    top_k=5
):

    query_text = application_to_text(
        application
    )


    query_embedding = (
        embedding_model.encode(
            [query_text],
            convert_to_numpy=True
        )
        .astype("float32")
    )


    faiss.normalize_L2(
        query_embedding
    )


    similarities, indices = (
        faiss_index.search(
            query_embedding,
            top_k
        )
    )


    retrieved_cases = []


    for similarity, index in zip(
        similarities[0],
        indices[0]
    ):

        case = metadata.iloc[
            index
        ]


        retrieved_cases.append(
            {
                "application_id":
                    case["application_id"],

                "similarity":
                    float(similarity),

                "decision":
                    case["decision"],

                "cibil_score":
                    case["cibil_score"],

                "business_vintage_months":
                    case[
                        "business_vintage_months"
                    ],

                "annual_revenue_lakhs":
                    case[
                        "annual_revenue_lakhs"
                    ],

                "existing_debt_lakhs":
                    case[
                        "existing_debt_lakhs"
                    ],

                "loan_amount_lakhs":
                    case[
                        "loan_amount_lakhs"
                    ],

                "sector":
                    case["sector"],

                "rejection_reasons":
                    case[
                        "rejection_reasons"
                    ],

                "loan_officer_notes":
                    case[
                        "loan_officer_notes"
                    ]
            }
        )


    return retrieved_cases


# =========================================================
# APPROVAL PROBABILITY HELPER
# =========================================================

def get_approval_probability(
    application
):

    _, probability = (
        predict_application(
            application
        )
    )

    return probability


# =========================================================
# COUNTERFACTUAL ENGINE
# =========================================================

def generate_counterfactuals(
    application
):

    original_probability = (
        get_approval_probability(
            application
        )
    )


    counterfactuals = []


    # -----------------------------------------------------
    # CIBIL scenarios
    # -----------------------------------------------------

    for new_cibil in [
        730,
        750,
        770,
        800
    ]:

        if (
            new_cibil
            <= application["cibil_score"]
        ):
            continue


        modified = (
            application.copy()
        )

        modified[
            "cibil_score"
        ] = new_cibil


        probability = (
            get_approval_probability(
                modified
            )
        )


        counterfactuals.append(
            {
                "change":
                    (
                        f"Increase CIBIL "
                        f"to {new_cibil}"
                    ),

                "probability":
                    probability,

                "improvement":
                    (
                        probability
                        - original_probability
                    )
            }
        )


    # -----------------------------------------------------
    # Debt scenarios
    # -----------------------------------------------------

    current_debt = application[
        "existing_debt_lakhs"
    ]


    for reduction in [
        0.10,
        0.25,
        0.50
    ]:

        new_debt = round(
            current_debt
            * (1 - reduction),
            2
        )


        modified = (
            application.copy()
        )


        modified[
            "existing_debt_lakhs"
        ] = new_debt


        probability = (
            get_approval_probability(
                modified
            )
        )


        counterfactuals.append(
            {
                "change":
                    (
                        f"Reduce debt "
                        f"to ₹{new_debt} lakh"
                    ),

                "probability":
                    probability,

                "improvement":
                    (
                        probability
                        - original_probability
                    )
            }
        )


    # -----------------------------------------------------
    # Business vintage scenarios
    # -----------------------------------------------------

    for vintage in [
        24,
        36,
        48
    ]:

        if (
            vintage
            <= application[
                "business_vintage_months"
            ]
        ):
            continue


        modified = (
            application.copy()
        )


        modified[
            "business_vintage_months"
        ] = vintage


        probability = (
            get_approval_probability(
                modified
            )
        )


        counterfactuals.append(
            {
                "change":
                    (
                        f"Increase business "
                        f"vintage to "
                        f"{vintage} months"
                    ),

                "probability":
                    probability,

                "improvement":
                    (
                        probability
                        - original_probability
                    )
            }
        )


    # -----------------------------------------------------
    # Loan amount scenarios
    # -----------------------------------------------------

    current_loan = application[
        "loan_amount_lakhs"
    ]


    for reduction in [
        0.10,
        0.25,
        0.50
    ]:

        new_loan = round(
            current_loan
            * (1 - reduction),
            2
        )


        modified = (
            application.copy()
        )


        modified[
            "loan_amount_lakhs"
        ] = new_loan


        probability = (
            get_approval_probability(
                modified
            )
        )


        counterfactuals.append(
            {
                "change":
                    (
                        f"Reduce requested "
                        f"loan to "
                        f"₹{new_loan} lakh"
                    ),

                "probability":
                    probability,

                "improvement":
                    (
                        probability
                        - original_probability
                    )
            }
        )


    counterfactuals.sort(
        key=lambda item:
            item["improvement"],
        reverse=True
    )


    return counterfactuals[:5]


# =========================================================
# BUILD RAG CONTEXT
# =========================================================

def build_context(
    similar_cases
):

    context_parts = []


    for number, case in enumerate(
        similar_cases,
        start=1
    ):

        text = (
            f"CASE {number}\n"
            f"Application ID: "
            f"{case['application_id']}\n"

            f"Similarity: "
            f"{case['similarity']:.3f}\n"

            f"Decision: "
            f"{case['decision']}\n"

            f"CIBIL: "
            f"{case['cibil_score']}\n"

            f"Business vintage: "
            f"{case['business_vintage_months']} months\n"

            f"Revenue: "
            f"₹{case['annual_revenue_lakhs']} lakh\n"

            f"Debt: "
            f"₹{case['existing_debt_lakhs']} lakh\n"

            f"Loan requested: "
            f"₹{case['loan_amount_lakhs']} lakh\n"

            f"Sector: "
            f"{case['sector']}\n"

            f"Rejection reasons: "
            f"{case['rejection_reasons']}\n"

            f"Officer notes: "
            f"{case['loan_officer_notes']}"
        )


        context_parts.append(
            text
        )


    return "\n\n".join(
        context_parts
    )


# =========================================================
# FORMAT COUNTERFACTUALS
# =========================================================

def format_counterfactuals(
    counterfactuals
):

    lines = []


    for number, item in enumerate(
        counterfactuals,
        start=1
    ):

        lines.append(
            (
                f"{number}. "
                f"{item['change']} -> "
                f"{item['probability'] * 100:.2f}% "
                f"approval probability "
                f"(change "
                f"{item['improvement'] * 100:+.2f}%)"
            )
        )


    return "\n".join(
        lines
    )


# =========================================================
# LLM GENERATION
# =========================================================

def generate_llm_response(
    application,
    prediction,
    approval_probability,
    similar_cases,
    counterfactuals
):

    context = build_context(
        similar_cases
    )


    counterfactual_text = (
        format_counterfactuals(
            counterfactuals
        )
    )


    prompt = f"""
You are LoanLens, an explainable MSME loan decision assistant.

You must only use the information supplied below.

Do not guarantee that a real bank will approve or reject a loan.

Clearly state that the result is a model-based estimate from synthetic prototype data.

CURRENT APPLICATION
{application_to_text(application)}

MODEL RESULT
Predicted decision: {prediction}
Approval probability: {approval_probability * 100:.2f}%

SIMILAR HISTORICAL CASES
{context}

COUNTERFACTUAL ANALYSIS
{counterfactual_text}

Prepare a concise report using exactly these sections:

1. Predicted Decision
2. Main Reasons
3. Similar Historical Cases
4. Recommended Improvements
5. Important Disclaimer

Use specific numbers wherever possible.
Do not invent bank thresholds or lending policies.
"""


    messages = [
        {
            "role": "system",
            "content":
                (
                    "You are a careful "
                    "financial decision "
                    "explanation assistant."
                )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


    formatted_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(
        llm.device
    )


    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=500,
        do_sample=False
    )


    new_tokens = generated_ids[
        :,
        model_inputs.input_ids.shape[1]:
    ]


    response = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True
    )


    return response


# =========================================================
# MAIN LOANLENS ANALYSIS
# =========================================================

def analyze_application(
    application
):

    prediction, approval_probability = (
        predict_application(
            application
        )
    )


    similar_cases = (
        retrieve_similar_cases(
            application,
            top_k=5
        )
    )


    counterfactuals = (
        generate_counterfactuals(
            application
        )
    )


    report = (
        generate_llm_response(
            application,
            prediction,
            approval_probability,
            similar_cases,
            counterfactuals
        )
    )


    return {
        "prediction":
            prediction,

        "approval_probability":
            approval_probability,

        "similar_cases":
            similar_cases,

        "counterfactuals":
            counterfactuals,

        "report":
            report
    }


# =========================================================
# TEST APPLICATION
# =========================================================

if __name__ == "__main__":

    test_application = {
        "business_vintage_months": 18,
        "annual_revenue_lakhs": 45.0,
        "existing_debt_lakhs": 12.0,
        "cibil_score": 720,
        "sector": "manufacturing",
        "loan_amount_lakhs": 15.0
    }


    print("\n==============================")
    print("RUNNING LOANLENS")
    print("==============================")


    result = analyze_application(
        test_application
    )


    print("\n==============================")
    print("MODEL RESULT")
    print("==============================")


    print(
        "Decision:",
        result["prediction"]
    )


    print(
        "Approval Probability:",
        f"{result['approval_probability'] * 100:.2f}%"
    )


    print("\n==============================")
    print("AI EXPLANATION")
    print("==============================")


    print(
        result["report"]
    )