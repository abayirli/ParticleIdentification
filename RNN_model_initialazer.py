# TRT RNN Code Model Initializer
# Reads the configuration file and arguments
# Reads the filelist for training and testing
# Defines the RNN model (trtLSTM)
# Defines the track generator  which will read the ROOT files and generate tracks

from lstm_helper_functions import *

#initialize the configurations
args = args()

num_hit_vars = len(hit_features)
num_track_vars = len(track_features)

#the model involves both hit variables and track variables - thus both of them are True
store_hits = True
store_tracks = True

show_plots = args.show_plots
report = args.report
prob = args.prob


device = torch.device("cuda:0" if (torch.cuda.is_available() and args.enable_cuda) else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device (CPU otherwise)
print(f"Compute engine: {device}")

enable_cuda_flag = True if args.enable_cuda == 'True' else False
args.cuda = enable_cuda_flag and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#read the normalization factors
with open(f"./mc_norm_factors.data", 'rb') as handle:  
    mc_norm_factors = pickle.load(handle)

#read the train and test files
filelist_train, filelist_test = file_dictionary_maker(file_paths, data_year)


# Create a SummaryWriter to write TensorBoard events locally
output_dir = dirpath = tempfile.mkdtemp()
writer = SummaryWriter(output_dir)
if(isTrain): print("Writing TensorBoard events locally to %s\n" % output_dir)

class trtLSTM(nn.Module):

    def __init__(self, num_hit_vars, num_track_vars, hidden_dim, batch_size):
        super(trtLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(num_hit_vars, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim + num_track_vars, 128)
        self.hidden2tag = nn.Linear(128, 1)
        
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = [torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)]
        return hidden

    def forward(self, packed_seq_batch, track_vars):
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(self.batch_size)
        hidden = [a.to(device) for a in hidden]
        output, (hn, cn) = self.lstm(packed_seq_batch, hidden)
        #pad the variable size input with zeros so that every element (track) in batch is the same length (same # of hits)
        padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=output[0].shape[0])

        #concatanate the LSTM output with track variables
        hit_track_combined = [torch.cat((padded_output[i][output_lens[i] - 1], track_vars[i].float()), -1) for i in range(args.batch_size)]
        #combine each output into a tensor which will be input to dense layer
        hit_track_combined = torch.stack(hit_track_combined, dim = 0)
        #push to the first dense layer
        dense = F.relu(self.dense(hit_track_combined))
        #push to the final output layer
        e_prob = torch.sigmoid(self.hidden2tag(dense))
        
        return e_prob


#initialize the model
hidden_dim = args.hidden_dim 
model = trtLSTM(num_hit_vars, num_track_vars, hidden_dim, args.batch_size)
optimizer = optim.Adam(model.parameters(), lr = args.lr)
model = model.float()
model.to(device) #send model to CUDA device if available
loss_function = nn.BCELoss()
print("Model defined!")

#if isTrain is set to False, the last model with the args.epoch number from model path is loaded (for analysis)
if(not isTrain):
    print("Model is loaded from saved model...")
    checkpoint = torch.load(f"./models/{model_name}_epoch{args.epochs}.pth", map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def trackGenerator(filelist, particle_type, scale = True):
    """Yields one track at a time for a given particle with scaling if set
    uses mean and standrd dev. values for scaling that are calculated from
    training set. If scale=False, it returns raw data values. unscaled
    """
    if(particle_type == "electron_mc"): label = 1
    if(particle_type == "electron_probes"): label = 1
    if(particle_type == "muon_mc"): label = 0
    if(particle_type == "muons"): label = 0
    for k in filelist:
        for arrays in uproot.iterate(k, particle_type, features): 
            for i in range(len(arrays[b'p'])): #loop over all the tracks in the root file
                e_eProbHT = arrays[b'eProbHT'][i]
                hit_array = []
                track = []
                if(store_tracks):
                    for f in track_features:
                        if(scale == False): 
                            (mean, std) = (0,1) 
                        else: 
                            (mean, std) = (mc_norm_factors[b"mean_"+f],mc_norm_factors[b"std_"+f])
                        track += [(arrays[f][i] - mean)/std ]
                if(store_hits):
                    if(arrays[b'hit_HTMB'][i][-1] == 99999): #check the hit arrays are padded with '99999' at the end
                        hit_length = list(arrays[b'hit_HTMB'][i]).index(99999)
                    else:
                        hit_length = len(arrays[b'hit_HTMB'][i])
                    for j in range(hit_length):
                        hit = []
                        for f in hit_features:
                            if(scale == False): 
                                (mean, std) = (0,1) 
                            else: 
                                (mean, std) = (mc_norm_factors[b"mean_"+f],mc_norm_factors[b"std_"+f])
                            hit += [(arrays[f][i][j] - mean)/std]
                        hit_array += [hit]
                yield [hit_array, track, e_eProbHT, label]



def create_batch(electron_gen, muon_gen, batch_size, prob):
    """Creates a mini-batch of tracks with a given batch_size consisting of eletrons and muons
    the proportion of electrons and muons controlled by the parameter prob
    input:
        electron_gen (generator): electron generator object for electron tracks
        muon_gen (generator): muon generator object for muon tracks
        batch_size (int): number of tracks in a given batch
        prob (float): parameter controlling the ratio of electrons to muons in a batch (1 = all electrons, 0 = all muons)
    returns:
        batch (list)
    """
    batch = []
    for m in range(args.batch_size):
        particle = next(electron_gen) if np.random.random() < prob else next(muon_gen)
        batch.append(particle)
    return batch

