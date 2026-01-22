Your task is to refactor the given codes and use it as an outer pip package instead of editable code repos. 

# Workflows: 
Each codes must have their corresponding elements in agents/FEATURES.md. 
If we only have raw agents/FEATURES.md and no codes, we will do "feature2codes". 
If we only have codes and no agents/FEATURES.md for this code, we will do "codes2feature".
IMPORTANT: codes to feature is necessary, since we can build new features based on existing features with "add_feature"

# Tools: 
here is the rules we should follow to make a feature to  a code.
<feature2codes>
Check for agents/FEATURES.md to understand the given features.
If we don't have codes and features, we will ask users' ideas to create or append new features in agents/FEATURES.md
Write simple code blocks with clear human-readable std and err log info. 
If everything worked well , we will ask if we can make this code into a script or helper or utils in our code repos.
After it worked, we will copy and clear our agents/FEATUREs.md and append them in docstrings. 
</feature2codes>

here is the rules we should follow to make a code into a feature. 
<codes2feature>
check for agents/FEATURES.md to check for existing features. 
check for our codes to understand the unique features in this code. 
We will refactor our codes into a feature list in agents/FEATURES.md and ask users if they want to append new features to it. 
</codes2feature>

<add_feature>
check if features and codes are parrellal. ask users problems to clarify the new features. 
IMPORTANT: make minimal changes to codes with file edit, it would be good to do just enough to cover NEW feature requests. 
</add_feature>