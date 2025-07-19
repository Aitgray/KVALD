# KVALD Project TODO List

This document outlines the necessary tasks to move the KVALD project from the prototype stage to a more robust and production-ready state.

## Up Next

### Data Pipeline

- [ ] **Real-world Data Integration:**
    - [ ] Compile a dataset of real driving footage.
    - [ ] Create a script to extract and preprocess this footage for the model.

### Model and Training

- [ ] **C++ Model Implementation:**
    - [ ] Translate the Python-based U-Net model to C++.
    - [ ] Integrate the C++ model with the data pipeline.
- [ ] **Training with Real Data:**
    - [ ] Train the C++ model using the real-world driving footage.
- [ ] **Refine Unsupervised Learning:**
    - [ ] Implement and refine the unsupervised learning loop with the defined heuristics.
- [ ] **CUDA Implementation:**
    - [ ] Investigate and implement CUDA support for PyTorch model training to leverage GPU acceleration.

---

## Completed

- [x] **`proof_of_concept.py` Refinements:**
    - [x] **Video Frame Streaming:** Replaced the frame extraction method with a more efficient video streaming approach.
    - [x] **Error Handling:** Implemented specific and informative error handling.
    - [x] **Data Type Consistency:** Decided on the optimal data type for grayscale conversion.
    - [x] **Feedback Verification (`verify_feedback`):** Implemented the `verify_feedback` function.
    - [x] **Final Output (`finalize_output`):** Completed the `finalize_output` function.
    - [x] **Multithreading:** Implemented multithreading to concurrently extract and process video frames.
- [x] **General:**
    - [x] **Code Documentation:** Added comprehensive docstrings to functions in `proof_of_concept.py`.
    - [x] **Testing:** Developed a suite of unit tests for `proof_of_concept.py`.
    - [x] **Configuration Management:** Reviewed and refined the use of `config.json` for managing parameters.
- [x] **Unsupervised Learning Heuristics:**
    - [x] Defined and implemented proper heuristics for the unsupervised learning phase.
- [x] **Performance Verification:**
    - [x] Created a script to verify the output of the proof of concept.