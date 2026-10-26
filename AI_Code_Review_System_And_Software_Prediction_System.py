import os
import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="AI Code Review & Defect Prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI-Powered Intelligent Code Review & Software Defect Prediction")
st.caption("Mini-project prototype | Python static analysis + optional Gemini review")

DEFAULT_CODE = '''def divide(a, b):
    result = a / b
    print(result)
    return result

def process_items(items):
    total = 0
    for i in range(len(items)):
        total += items[i]
    return total
'''


ML_FEATURES = [
    "loc", "functions", "classes", "imports", "try_except",
    "cyclomatic_complexity", "eval_calls", "bare_except",
    "long_lines", "missing_docstrings",
]

def extract_code_features(source: str) -> dict:
    """Extract simple, explainable features from Python source code."""
    lines = source.splitlines()
    features = {
        "loc": sum(1 for line in lines if line.strip()),
        "functions": 0, "classes": 0, "imports": 0, "try_except": 0,
        "cyclomatic_complexity": 1, "eval_calls": 0, "bare_except": 0,
        "long_lines": sum(1 for line in lines if len(line) > 100),
        "missing_docstrings": 0,
    }
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Keep the predictor usable for invalid code, while static analysis reports syntax error.
        features["cyclomatic_complexity"] = 10
        features["missing_docstrings"] = 1
        return features

    decision_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.And, ast.Or, ast.ExceptHandler, ast.comprehension)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features["functions"] += 1
            if not ast.get_docstring(node):
                features["missing_docstrings"] += 1
        elif isinstance(node, ast.ClassDef):
            features["classes"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            features["imports"] += 1
        elif isinstance(node, ast.Try):
            features["try_except"] += 1
        elif isinstance(node, decision_nodes):
            features["cyclomatic_complexity"] += 1
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            features["bare_except"] += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            features["eval_calls"] += 1
    return features

def make_demo_dataset(n=500, seed=42):
    """Create synthetic demonstration data; NOT a real-world defect benchmark."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "loc": rng.integers(5, 500, n),
        "functions": rng.integers(0, 30, n),
        "classes": rng.integers(0, 10, n),
        "imports": rng.integers(0, 25, n),
        "try_except": rng.integers(0, 12, n),
        "cyclomatic_complexity": rng.integers(1, 45, n),
        "eval_calls": rng.integers(0, 3, n),
        "bare_except": rng.integers(0, 4, n),
        "long_lines": rng.integers(0, 15, n),
        "missing_docstrings": rng.integers(0, 20, n),
    })
    # Synthetic label rule only for a working demo, not empirical defect truth.
    score = (data.loc[:, ["loc", "cyclomatic_complexity", "eval_calls", "bare_except", "long_lines", "missing_docstrings"]].to_numpy() * np.array([0.002, 0.035, 0.9, 0.45, 0.035, 0.025])).sum(axis=1)
    probability = 1 / (1 + np.exp(-(score - 1.5)))
    data["defective"] = (rng.random(n) < probability).astype(int)
    return data

def train_defect_model(dataset: pd.DataFrame):
    required = set(ML_FEATURES + ["defective"])
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError("Dataset mein ye columns missing hain: " + ", ".join(missing))
    data = dataset[ML_FEATURES + ["defective"]].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    data = data[data["defective"].isin([0, 1])]
    if len(data) < 10 or data["defective"].nunique() < 2:
        raise ValueError("Kam se kam 10 valid rows aur dono labels (0 aur 1) zaroori hain.")
    X, y = data[ML_FEATURES], data["defective"].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Test rows": len(y_test),
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
        "F1-score": round(f1_score(y_test, pred, zero_division=0), 3),
    }
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    importance = pd.DataFrame({"Feature": ML_FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    return model, metrics, cm, importance, len(data)


def get_gemini_review(source: str) -> str:
    """Ask Gemini to review code; return a readable message on API failures."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "Gemini review skipped: GEMINI_API_KEY environment variable nahi mili. "
            "Static analysis neeche phir bhi chalega."
        )

    # Override this value with GEMINI_MODEL if a different model is available to your API key.
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=120000),
        )
        response = client.models.generate_content(
    model=model_name,
    contents=(
        "Review this Python code for likely bugs, security issues, and improvements. "
        "Explain findings in simple language. Do not claim a bug is certain unless "
        "the code demonstrates it. Give concise actionable suggestions.\n\n"
        f"Python code:\n```python\n{source}\n```"
      ),
    )
        return response.text or "Gemini ne koi text response nahi diya."
    except Exception as exc:
        error_text = str(exc)
        lower = error_text.lower()

        if "429" in error_text or "resource_exhausted" in lower or "quota" in lower:
            return (
                "Gemini quota abhi exhausted hai. Free-tier quota reset hone ya quota "
                "available hone par dobara try karein. Is wajah se static analysis band "
                "nahi hota. Repeated clicks se quota wapas nahi aayegi."
            )
        if "503" in error_text or "unavailable" in lower:
            return (
                "Gemini service filhaal unavailable/busy hai. Thodi der baad try karein. "
                "Static analysis neeche available hai."
            )
        if "404" in error_text or "not found" in lower:
            return (
                f"Gemini model '{model_name}' available nahi lag raha. Google AI Studio "
                "mein apne API key ke liye supported model check karein, phir GEMINI_MODEL "
                "environment variable set karein."
            )
        if "timeout" in lower or "timed out" in lower:
            return (
                "Gemini request timeout ho gayi. Internet connection check karein aur "
                "baad mein dobara try karein. Static analysis phir bhi available hai."
            )
        return f"Gemini review nahi ho saka: {error_text[:700]}"


def analyse_code(source: str):
    """Run transparent AST-based checks; this is not a trained ML prediction."""
    issues = []
    metrics = {
        "Lines of code": len([line for line in source.splitlines() if line.strip()]),
        "Functions": 0,
        "Classes": 0,
        "Imports": 0,
        "Try/except blocks": 0,
    }
    score_penalty = 0

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        issue = [{
            "Severity": "High",
            "Issue": "Syntax error",
            "Details": f"{exc.msg} (line {exc.lineno})",
            "Suggestion": "Syntax fix karein, phir analysis dobara run karein.",
        }]
        return issue, metrics, 100

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics["Functions"] += 1
            if len(node.args.args) > 5:
                issues.append({
                    "Severity": "Medium",
                    "Issue": "Many function parameters",
                    "Details": f"'{node.name}' has more than 5 parameters.",
                    "Suggestion": "Related parameters ko data class ya object mein group karein.",
                })
                score_penalty += 8

            if len(node.body) > 25:
                issues.append({
                    "Severity": "Medium",
                    "Issue": "Long function",
                    "Details": f"'{node.name}' has a large top-level body.",
                    "Suggestion": "Function ko chhote, single-purpose functions mein split karein.",
                })
                score_penalty += 8

            if not ast.get_docstring(node):
                issues.append({
                    "Severity": "Low",
                    "Issue": "Missing function docstring",
                    "Details": f"'{node.name}' has no docstring.",
                    "Suggestion": "Function ka purpose, inputs aur output batane wala docstring add karein.",
                })
                score_penalty += 2

        elif isinstance(node, ast.ClassDef):
            metrics["Classes"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            metrics["Imports"] += 1
        elif isinstance(node, ast.Try):
            metrics["Try/except blocks"] += 1
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "Severity": "Medium",
                "Issue": "Bare except clause",
                "Details": "Bare except unexpected errors ko hide kar sakta hai.",
                "Suggestion": "Specific exception types catch karein.",
            })
            score_penalty += 10
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "eval":
                issues.append({
                    "Severity": "High",
                    "Issue": "Use of eval()",
                    "Details": "eval() dynamically supplied Python expressions execute kar sakta hai.",
                    "Suggestion": "eval() se bachein; safe parser ya explicit logic use karein.",
                })
                score_penalty += 20
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                issues.append({
                    "Severity": "High",
                    "Issue": "Division by constant zero",
                    "Details": "Code literal value 0 se divide karta hai.",
                    "Suggestion": "Denominator validate karein aur zero division prevent karein.",
                })
                score_penalty += 25

    for line_number, line in enumerate(source.splitlines(), start=1):
        if len(line) > 100:
            issues.append({
                "Severity": "Low",
                "Issue": "Long line",
                "Details": f"Line {line_number} contains {len(line)} characters.",
                "Suggestion": "Line ko chhote readable expressions mein break karein.",
            })
            score_penalty += 1
        if "TODO" in line or "FIXME" in line:
            issues.append({
                "Severity": "Low",
                "Issue": "Unresolved marker",
                "Details": f"Line {line_number} contains TODO/FIXME.",
                "Suggestion": "Release se pehle marker resolve karein.",
            })
            score_penalty += 2

    if not issues:
        issues.append({
            "Severity": "Info",
            "Issue": "No configured rules triggered",
            "Details": "Current rules ne koi configured issue detect nahi kiya.",
            "Suggestion": "Iska matlab code defect-free hona prove nahi hota; tests bhi run karein.",
        })

    return issues, metrics, min(100, score_penalty)


with st.sidebar:
    st.header("About this prototype")
    st.write("Python AST-based static checks aur transparent heuristic scoring.")
    st.warning(
        "Risk score demo heuristic hai; yeh trained ML model ya defect probability nahi hai."
    )
    st.write("Workflow: code paste karein → analyse karein → findings review karein → tests run karein.")
    use_gemini = st.checkbox(
        "Gemini AI review run karein (API quota use hogi)",
        value=False,
        help="Quota exhausted ho toh checkbox untick rakhein. Static analysis free mein chalega.",
    )

code = st.text_area("Paste Python code", value=DEFAULT_CODE, height=300)

if st.button("Analyse Code", type="primary"):
    findings, metrics, risk = analyse_code(code)

    st.subheader("🤖 Gemini AI Review")
    if use_gemini:
        with st.spinner("Gemini response ka wait ho raha hai..."):
            st.markdown(get_gemini_review(code))
    else:
        st.info(
            "Gemini review skip kiya gaya hai. Static analysis run ho gaya. "
            "Gemini use karna ho toh sidebar checkbox select karein; isse API quota use hogi."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated heuristic risk", f"{risk}/100")
    col2.metric("Issues / findings", len(findings))
    col3.metric("Lines of code", metrics["Lines of code"])

    st.subheader("Code metrics")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

    st.subheader("Review findings")
    st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)

    if risk >= 25:
        st.error("Higher heuristic risk: findings review karein aur tests add karein.")
    elif risk >= 10:
        st.warning("Moderate heuristic risk: findings inspect karein aur edge cases test karein.")
    else:
        st.success(
            "Current rules ke hisaab se low heuristic score. Iska matlab code defect-free "
            "hona nahi hai."
        )

    csv_report = pd.DataFrame(findings).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download findings as CSV",
        data=csv_report,
        file_name="code_review_report.csv",
        mime="text/csv",
    )
    st.info(
        "Next steps: unit tests run karein, edge cases check karein, aur findings manually validate karein. "
        "ML model train karne ke liye labelled defect dataset alag se chahiye."
    )


# ---------------- Software Defect Prediction (ML) ----------------
st.divider()
st.header("🧪 Software Defect Prediction — Machine Learning")
st.write(
    "Yeh section code metrics se defective / non-defective class predict karta hai. "
    "Real evaluation ke liye labelled software defect dataset upload karein."
)

with st.expander("Dataset format aur feature definitions", expanded=False):
    st.markdown(
        "CSV mein ye columns hone chahiye: `loc`, `functions`, `classes`, `imports`, "
        "`try_except`, `cyclomatic_complexity`, `eval_calls`, `bare_except`, `long_lines`, "
        "`missing_docstrings`, `defective`. Label `defective` mein 1 = defective aur 0 = non-defective. "
        "Feature values software modules ki metrics honi chahiye; label actual known defect status hona chahiye."
    )
    example = pd.DataFrame([{**{f: 0 for f in ML_FEATURES}, "loc": 40, "functions": 3, "cyclomatic_complexity": 4, "defective": 0}, {**{f: 0 for f in ML_FEATURES}, "loc": 260, "functions": 12, "cyclomatic_complexity": 22, "eval_calls": 1, "defective": 1}])
    st.download_button("Download example CSV template", example.to_csv(index=False).encode("utf-8"), "defect_dataset_template.csv", "text/csv")

uploaded_dataset = st.file_uploader("Upload labelled defect dataset (CSV)", type=["csv"], key="defect_dataset")
use_demo_data = st.checkbox("Use synthetic demo dataset (sirf demonstration ke liye)", value=uploaded_dataset is None, help="Synthetic labels real software defect data nahi hain; report mein ise real-world accuracy na batayein.")

dataset = None
dataset_label = ""
if uploaded_dataset is not None:
    try:
        dataset = pd.read_csv(uploaded_dataset)
        dataset_label = "Uploaded labelled dataset"
    except Exception as exc:
        st.error(f"CSV read nahi ho saki: {exc}")
elif use_demo_data:
    dataset = make_demo_dataset()
    dataset_label = "Synthetic demonstration dataset (500 generated rows)"

if dataset is not None:
    st.caption(f"Training source: {dataset_label}")
    try:
        model, ml_metrics, cm, importance, row_count = train_defect_model(dataset)
        st.success(f"Model train ho gaya. Valid dataset rows: {row_count}")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{ml_metrics['Accuracy']:.1%}")
        metric_cols[1].metric("Precision", f"{ml_metrics['Precision']:.1%}")
        metric_cols[2].metric("Recall", f"{ml_metrics['Recall']:.1%}")
        metric_cols[3].metric("F1-score", f"{ml_metrics['F1-score']:.1%}")
        st.caption(f"Hold-out test set: {ml_metrics['Test rows']} rows. Metrics dataset aur split par depend karte hain.")
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual non-defective (0)", "Actual defective (1)"], columns=["Predicted non-defective (0)", "Predicted defective (1)"]), use_container_width=True)
        st.subheader("Feature importance")
        st.dataframe(importance, use_container_width=True, hide_index=True)
        current_features = extract_code_features(code)
        prediction = int(model.predict(pd.DataFrame([current_features], columns=ML_FEATURES))[0])
        probabilities = model.predict_proba(pd.DataFrame([current_features], columns=ML_FEATURES))[0]
        defect_index = list(model.classes_).index(1)
        probability = float(probabilities[defect_index])
        st.subheader("Prediction for the code above")
        st.write(pd.DataFrame([current_features]))
        if prediction == 1:
            st.warning(f"ML model prediction: Potentially defective (model score: {probability:.1%}). Isse confirmed bug na samjhein.")
        else:
            st.success(f"ML model prediction: Likely non-defective (defect-class model score: {probability:.1%}). Yeh defect-free hone ka proof nahi hai.")
        if uploaded_dataset is None:
            st.warning("Important: yeh model synthetic labels par trained hai. Iske scores sirf pipeline demonstration ke liye hain, real-world software quality ka evidence nahi. Submission se pehle possible ho toh actual labelled dataset use karein.")
        st.download_button("Download model evaluation metrics (CSV)", pd.DataFrame([ml_metrics]).to_csv(index=False).encode("utf-8"), "defect_model_metrics.csv", "text/csv")
    except Exception as exc:
        st.error(f"ML model train nahi ho saka: {exc}")
