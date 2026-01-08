<h1>Honey Adulteration Detection Using Spectroscopic Analysis</h1>

<h1>Overview : </h1>
This project detects pure vs adulterated honey using spectroscopic data and machine learning.
Spectroscopy captures how a sample interacts with light across many wavelengths, revealing its chemical composition, which helps identify food fraud.

<h1>Adulterants Considered:</h1>
High-Fructose Corn Syrup (HFCS)
Rice syrup
Maltose syrup

These adulterants alter the honey’s spectral signature, enabling detection.
Techniques: FTIR / UV-Vis / Raman spectroscopy
Features: Wavenumbers (230–1021 nm)
Values: Raman intensities
Target: Pure honey / Adulterated honey

<h1>Methodology:</h1>

Model: Support Vector Machine (SVM)
Kernel: RBF (Radial Basis Function)

*SVM Works well with small, high-dimensional datasets
*PCA: Used for dimensionality reduction and visualization
<img width="418" height="315" alt="image" src="https://github.com/user-attachments/assets/c6b002d6-f4ec-4419-86b2-fc3255f69f06" />
<img width="438" height="257" alt="image" src="https://github.com/user-attachments/assets/c5227bcd-2080-4594-96cd-08a99403a3c7" />

RBF-SVM captures nonlinear chemical patterns by mapping data into a high-dimensional space.

<h1>Results :</h1>
High classification accuracy on test data - 99.61%
Clear separation between pure and adulterated honey samples
Demonstrates the effectiveness of spectroscopy + ML for food authentication


