# Credit-Risk-Probability-Model-for-Alternative-Data
An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

#Task-1
1-The Basel II Capital Accord creates a strong need for interpretable and well-documented risk models because they directly determine regulatory capital requirements. Under the Advanced Internal Ratings-Based (A-IRB) approach, banks use internal models to estimate PD, LGD, and EAD, which must be clearly explained and justified to regulators.

As a result, model transparency and auditability are critical. Institutions often prefer logistic regression and credit scorecards because they are highly interpretable and regulator-friendly. Although advanced machine learning models can improve accuracy, their “black box” nature limits their use in regulated environments.

Basel II also requires thorough documentation, monitoring, and periodic recalibration of models to ensure accountability, manage model risk, and maintain ongoing regulatory compliance.

2-A proxy variable for default is necessary because credit scoring models require a binary response variable to estimate the Probability of Default (PD). When an explicit default label is unavailable or delayed, a proxy is constructed from observable data to represent whether a borrower defaulted or not.

However, using an imperfect proxy introduces business and modeling risks. If the proxy does not accurately reflect true default behavior, the model may misestimate risk, leading to higher default rates and poor lending decisions. It can also worsen sample bias, causing the model to underestimate risk. A noisy proxy weakens model interpretability, accelerates model decay, and may create regulatory compliance issues. Therefore, proxy variables must be carefully designed, validated, and continuously monitored.

3-The trade-off between simple, interpretable models (e.g., Logistic Regression or Credit Scorecards) and complex, high-performance models (e.g., GBM or Random Forests) lies in balancing predictive accuracy with transparency and regulatory compliance.

Simple models are highly interpretable, computationally efficient, and regulator-friendly, making them well suited for regulated environments such as Basel II, where models are used to estimate Probability of Default (PD) and must be easily explained, justified, and audited. Credit scorecards, in particular, allow clear insight into how borrower characteristics affect risk, supporting Explainable AI (XAI) requirements.

In contrast, complex machine learning models can capture non-linear relationships and subtle data patterns, often delivering higher predictive accuracy. However, they are typically less transparent, harder to explain, more computationally demanding, and more prone to overfitting if not carefully tuned.

In practice, financial institutions must balance these competing goals—often accepting lower accuracy in exchange for interpretability, stability, and regulatory acceptance, especially when models directly influence lending decisions and capital requirements.