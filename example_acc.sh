cd Bernstein_Polynomial_Approximation &&\
make name=acc_model &&\
python3 acc_ea.py --filename acc_model --error_bound 0.1
