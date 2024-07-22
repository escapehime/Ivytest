#enviroment setup
import pandas as pd
import numpy as np


directory_path = '/Users/yihsinchu/Downloads/test'

ABCDmatrix_file_name = 'ABCDmatrix.csv'

ABCDmatrix_full_path = f'{directory_path}/{ABCDmatrix_file_name}'
df_ABCDmatrix = pd.read_csv(ABCDmatrix_full_path, sep=',', names=['A', 'B', 'C', 'D', 'E', 'F'], skiprows=1)


# Directly access the values at E2 and F2
E0_value = df_ABCDmatrix.at[0, 'E']  # Accessing the value at row index 1, column 'E'
F0_value = df_ABCDmatrix.at[0, 'F']  # Accessing the value at row index 1, column 'F'

# Create the Incident_matrix with E2 as the first row and F2 as the second row
Incident_matrix = np.array([[E0_value], [F0_value]])

# Define last_E_value before the loop
last_E_value = None

#define process
def definition_process(row):
    B_value = row['B']
    C_value = row['C']
    D_value = row['D']
    E_value = row['E']
    ONE = int(1)
    ZERO =int(0)
    

    # Initialize Process_matrix to avoid UnboundLocalError
    Process_matrix = np.zeros((2, 2))  # Default to a 2x2 matrix filled with zeros

    if B_value == 'None':
         if B_value == 'None' and last_E_value is not None:  # Check if last_E_value is available
             Delta = E_value - last_E_value
         else:
             Delta = E_value - E0_value  # Initial calculation of Delta
    elif B_value == 'FlatLens':
        Nv = ONE / C_value
        Process_matrix = np.array([[ONE, ZERO], [ZERO, Nv]])
    elif B_value == 'Curvedlens':
        NvRv = ((ONE - C_value) / D_value * C_value)
        Nv = ONE / C_value
        Process_matrix = np.array([[ONE, ZERO], [NvRv, Nv]])
    else:
        print('Lens type not included')
    
    return Process_matrix

Process_matrices = []
for index, row in df_ABCDmatrix.iloc[2:].iterrows():
    Process_matrix = definition_process(row)
    if Process_matrix is not None:
        Process_matrices.append(Process_matrix)
        last_E_value = row['E']
        # Update last_E_value after successful processing
    else:
        print('hi')


#Calculation Process
Emergent_matrices = []
for i, Process_matrix in enumerate(Process_matrices):
    Emergent_matrix = np.dot(Process_matrix, Incident_matrix)
    Emergent_matrices.append(Emergent_matrix)
    # Update Incident_matrix for the next iteration
    Incident_matrix = Emergent_matrix


#End process
# Prepare the result DataFrame
result_df = pd.DataFrame({'Xe': [Emergent_matrices[-1][0, 0]], 'Xetheta': [Emergent_matrices[-1][1, 0]]})

# Create a new Excel writer object
writer = pd.ExcelWriter('final_result.csv', engine='openpyxl')

# Save the DataFrames to new CSV files
result_df.to_csv('final_result.csv', index=False, startrow=1, header=['Xe', 'Xetheta'])
