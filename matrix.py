#enviroment setup
import pandas as pd
import numpy as np

directory_path = '/Users/yihsinchu/Downloads/test'

# Read the CSV file, skipping the first row
df = pd.read_csv('ABCDmatrix.csv')

# Create the Incident_matrix from columns E and F, starting from row 2
Incident_matrix = df.iloc[0, 4:6].values.reshape(2, 1)
print(Incident_matrix)


# Function to create the Definition_Matrix based on the lens type
def create_definition_matrix(row, prev_row):
    lens_type = row.iloc[1]  # Assuming lens type is in the second column
    if pd.isna(lens_type):
        raise ValueError("Unexpected empty lens type")
    elif lens_type == 'Air':
        Delta = row.iloc[4] - prev_row.iloc[4]  # 5th column of current row minus 5th column of previous row
        print(Delta)
        return np.array([[1, Delta], [0, 1]])
    elif lens_type == 'FlatLens':
        Nv = 1 / row.iloc[2]  # Assuming C is the third column
        print(Nv)
        return np.array([[1, 0], [0, Nv]])
    elif lens_type == 'Curvedlens':
        NvRv = (1 - row.iloc[2]) / (row.iloc[3] * row.iloc[2])  # Assuming C is 3rd and D is 4th column
        Nv = 1 / row.iloc[2]
        print(NvRv, Nv)
        return np.array([[1, 0], [NvRv, Nv]])
    else:
        raise ValueError(f'Lens type not included: {lens_type}')

# Initialize the Emergent_matrix
Emergent_matrix = Incident_matrix

# Process each row starting from the row after Incident_matrix
for index in range(1, len(df)):
    row = df.iloc[index]
    prev_row = df.iloc[index - 1]
    
    # Check if the first column is empty
    if pd.isna(row.iloc[0]):
        print(f"Reached end of data at row {index + 2}")  # +2 because we skipped the first row and 0-indexing
        break
    
    # Definition process
    try:
        Definition_matrix = create_definition_matrix(row, prev_row)
    except Exception as e:
        print(f"Error processing row {index + 2}: {e}")  # +2 for the same reason as above
        break
    
    # Calculation process
    Emergent_matrix = np.dot(Definition_matrix, Emergent_matrix)

    print(f"Processed row {index + 2}, {Emergent_matrix}")  # Add this line for debugging

# Create the result DataFrame
print(Emergent_matrix) #test the result
result_df = pd.DataFrame({
    'X_emergent': [Emergent_matrix[0, 0]],
    'Xtheta_emergent': [Emergent_matrix[1, 0]]})

# Save the result to a new CSV file
result_df.to_csv('result.csv', index=False)

print("Processing complete. Results saved to 'result.csv'.")
