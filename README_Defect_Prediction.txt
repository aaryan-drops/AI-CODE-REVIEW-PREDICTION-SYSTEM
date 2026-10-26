AI-Powered Code Review and Software Defect Prediction System

1. Install Python packages:
   pip install -r requirements_defect_prediction.txt

2. Run the Streamlit app:
   streamlit run AI_Code_Review_System_Updated-1.py

3. Gemini setup:
   Set GEMINI_API_KEY in your environment. Optional: set GEMINI_MODEL to a model supported by your API key.

4. ML dataset:
   The app accepts a labelled CSV with these exact columns:
   loc, functions, classes, imports, try_except, cyclomatic_complexity,
   eval_calls, bare_except, long_lines, missing_docstrings, defective

   defective: 1 = defective, 0 = non-defective.
   Use actual known defect labels from software modules for a meaningful evaluation.

5. Synthetic demo warning:
   The built-in synthetic dataset is generated for demonstrating the training/evaluation pipeline only.
   Do not present synthetic-data metrics as real-world performance. For the final report, state clearly
   which dataset was used and report its provenance and limitations.
