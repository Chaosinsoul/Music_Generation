# How to generate music and heat maps using our default setup.
Run the python code stored in MusicGeneration-and-Heatmaps.py.
The code is split into 4 sections:
# Parameters
# Build and Training Model
# Generate Music
# Print heat maps
ctrl-f these strings to find the start and end point of each section.
Run each section in sequence. Only rerun # Build and Training Model if you
change and run the parameters section first.

To reproduce any of our results alter the parameters in the # Parameters section

To change the dropout find the MusicGenerator model in the # Build and Training Model
section and change the line "self.dropout = nn.Dropout(p=0.0)"

To change the optimizer ctrl-f #create model and look 3 lines down to locate and uncomment an optimizer function:
optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum = moment, nesterov = True)
#optimizer = optim.RMSprop(model.parameters())
#optimizer = optim.Adagrad(model.parameters(), lr=learningRate, lr_decay=0, weight_decay=0)
*Some parameters become useless when you switch from SGD
