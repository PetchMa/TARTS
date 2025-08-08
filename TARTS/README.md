# Rubin Realtime Neural Active Optics System
To train a model from scratch we run the following procedures
1. Train base `wavenet` and `alignnet` on augmented archival simulation. This will serve as pretraining
2. Generate data using the `alignnet` to center new fullframe crops from smaller/more expensive/more realistic simulation
3. Train/Finetune `wavenet` on the realigned cropped images on the smaller realistic simulation (one zernike per crop)
4. Generate sets of estimations of Zernikes using the finetuned `wavenet` on the full frame. (full frame with multiple zernike est.)
5. Train `aggregatornet` to correct the mean of the `wavenet` estimates.
6. Take all model weights and put them into the `NeuralActiveOptic` model
7. Finetune `NeuralActiveOptic` for real data.
8. Optimize model performance, quantisation, layer fusion, pruning etc...
9. alpha = 500
