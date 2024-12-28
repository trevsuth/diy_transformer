```mermaid
flowchart TD
    %% Input Section
    A[Inputs] --> B[Input Embedding]
    B --> C[Positional Encoding]
    C --> D1["Add & Norm"]
    D1 --> E1["Multi-Head Attention"]
    E1 --> F1["Add & Norm"]
    F1 --> G1["Feed Forward"]
    G1 --> H1["Add & Norm"]
    H1 --> K1["N x Layers"]

    %% Output Section
    O[Outputs (shifted right)] --> P[Output Embedding]
    P --> Q[Positional Encoding]
    Q --> R1["Add & Norm"]
    R1 --> S1["Masked Multi-Head Attention"]
    S1 --> T1["Add & Norm"]
    T1 --> U1["Multi-Head Attention"]
    U1 --> V1["Add & Norm"]
    V1 --> W1["Feed Forward"]
    W1 --> X1["Add & Norm"]
    X1 --> Y1["N x Layers"]

    %% Connecting Input and Output Layers
    K1 --> U1

    %% Final Linear and Softmax
    Y1 --> Z[Linear]
    Z --> AA[Softmax]
    AA --> AB[Output Probabilities]

```