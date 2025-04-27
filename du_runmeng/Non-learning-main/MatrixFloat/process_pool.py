from utils_float import CSP_PA, Precision, num_of_client
import numpy as np
N = num_of_client


def calculate_parameters(received_parameters, A_Matrix_):
    global Wh, Wo
    index, to_Wo, to_Wh_1, to_Wh_2 = received_parameters
    # print(f"Recieved {index}")
    
    # de_to_Wo = (pailliar_dec(to_Wo)) / (Precision ** 2)
    # de_to_Wh1 = (to_Wh_1 @ A_Matrix_[index].T) / (Precision)
    # de_to_Wh2 =  pailliar_dec(to_Wh_2)
    de_to_Wo = (CSP_PA.decryMatrix(to_Wo)) / (Precision ** 2)
    de_to_Wh1 = (to_Wh_1 @ A_Matrix_.T)
    # TODO Decrypt C2 with CSP. Remember to decrypt in CSP.
    
    de_to_Wh2 =  CSP_PA.decryMatrix(to_Wh_2) / (Precision ** 2)
    
    delta_Wo = de_to_Wo / N
    delta_Wh = (de_to_Wh1 * np.repeat(de_to_Wh2.T, de_to_Wh1.shape[1], axis=1)) / N
    # delta_Wh = ((to_Wh_1.T @ to_Wh_2).T) / N
    
    return delta_Wh, delta_Wo 