# RULNet
This repository contains the code and documentation for predicting the Remaining Useful Life (RUL) of lithium-ion batteries using advanced machine learning techniques. Our approach leverages various flavors of Recurrent Neural Networks (RNNs), including LSTM, BiLSTM, GRU, and BiGRU, to model the degradation trajectories of lithium-ion batteries. By ensembling these RNN models, we aim to overcome individual model shortcomings and improve prediction accuracy.

The dataset used in this project was obtained from the NASA Ames Prognostics Center of Excellence. It includes data from lithium-ion batteries subjected to different operational profiles, capturing various parameters such as voltage, current, temperature, and impedance over time. The batteries were cycled until they reached end-of-life (EOL) criteria, defined by a 30% fade in rated capacity.
