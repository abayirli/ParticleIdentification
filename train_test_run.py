#Train Notes
#LSTM(50) + 1Dense(100) 1M500K 100 epoch with batch_size = 64, lr  = 0.0001 Adam + ReLU added - RF selected features

from RNN_model_initialazer import *


#set the random seed for reproducibility
np.random.seed(args.seed)


#Create directory to store models and plots
path = os.getcwd()
work_dir = path
print(path)

try:
    os.mkdir(path + "/plots")
    os.mkdir(path + "/models")  
except OSError:
    print ("Creation of models/plots the directory failed")
else:
    print ("Successfully created the models/plots directory")


prob = args.prob #balance of the train/test/val data sets (0.5 = balanced)


#Initialize the generators to be used to create test and validation sets
electron_val = trackGenerator(filelist_test['electron_mc'], particle_type = "electron_mc")
muon_val = trackGenerator(filelist_test['muon_mc'], particle_type = "muon_mc")

test_set = []
test_size = int(args.test_size/args.batch_size)
for k in range(test_size):
     test_batch = create_batch(electron_val, muon_val, args.batch_size, prob)
     test_set.append(test_batch)

print("Test data created!")


#Create validation set
val_set = []
val_size = int(args.val_size/args.batch_size)
for k in range(val_size):
     val_batch = create_batch(electron_val, muon_val, args.batch_size, prob)
     val_set.append(val_batch)

print("Validation data created!")

#create test DATA set
electron_val_data = trackGenerator(filelist_test['electron_probes2016'], particle_type="electron_probes")
muon_val_data = trackGenerator(filelist_test['muons2016'], particle_type = "muons")

test_set_data = []
test_size = int(args.test_size/args.batch_size)
for k in range(test_size):
	test_batch = create_batch(electron_val_data, muon_val_data, args.batch_size, prob)
	test_set_data.append(test_batch)
print("DATA test set created!")


def train(epoch):
    """Train function runs for each epoch and for a given set of batches of training data
    it updates the parameters of the model. It loops over the batches and for each batch
    it calculates the output and logs the corresponding loss
    """
    model.train()
    start = time.time()
    electron_train = trackGenerator(filelist_train['electron_mc'], particle_type = "electron_mc")
    muon_train = trackGenerator(filelist_train['muon_mc'], particle_type = "muon_mc")
    start = time.time()
    train_loss = 0
    start = time.time()
    
    for k in range(int(args.train_size/args.batch_size)):
        model.zero_grad()
        tracks = create_batch(electron_train, muon_train, args.batch_size, prob)
        #a dictionary to keep track of each variable-length track and their labels
        #tracks[i][0] = hit-sequence associated to the track
        #tracks[i][3] = label (1: electron, 0: muon)
        #tracks[i][1] = track variables associated to the track
        seq_dict  = [(torch.tensor(tracks[i][0]), torch.tensor(tracks[i][3]), torch.tensor(tracks[i][1])) for i in range(args.batch_size)]
        #sort the tracks in decreasing number of hits to prepare for padding
        seq_batch_sorted = sorted(seq_dict, key=lambda x: len(x[0]), reverse=True)
        #get the labels of the sorted tracks       
        label_batch = [seq_batch_sorted[i][1].float().to(device) for i in range(args.batch_size)]
        #get the hit sequences of the sorted tracks
        seq_batch = [seq_batch_sorted[i][0] for i in range(args.batch_size)]
        #get the sequence lengths of each track in the batch
        seq_lens = [seq_batch[i].shape[0] for i in range(args.batch_size)]
        #get the associated track variables for each track in the batch
        track_batch = [seq_batch_sorted[i][2].float().to(device) for i in range(args.batch_size)]

        #pad the hit sequences with 0's to make the number of hits in each track to be equal       
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_batch, batch_first=True)
        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).float().to(device)


        out = model(packed_seq_batch, track_batch)
        loss = loss_function(out, torch.stack(label_batch, 0).view(args.batch_size, 1)).to(device)

        train_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        if k % args.log_interval == args.log_interval - 1:
            if(report): print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(epoch, 100. * k / (args.train_size/args.batch_size), train_loss/(k+1)))
            step = epoch * args.train_size + k
            log_scalar('train_loss', train_loss/(k+1), step)
            #model.log_weights(step) #comment out if you dont want the weights to be stored for further analysis

    if(args.statesave):
        state = { 'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'train_size': args.train_size,
              'comment: ': f'{args.epochs} epoch {args.train_size} MC training - lr = {args.lr}'}
        filename = f'models/{model_name}_epoch{epoch}.pth'
        torch.save(state, filename)
    print(f"Time for Epoch {epoch}: {timeSince(start)}")

def validate(epoch):
    """Validation function runs for each epoch and for a given set of batches of validation data
    it loops over them and for each batch it calculates the output and logs the corresponding loss
    """
    model.eval()
    y_scores = []
    yval = []
    val_eProbHT = []
    val_loss_list = []
    val_loss = 0
    with torch.no_grad():
        for tracks in val_set:
            seq_dict  = [(torch.tensor(tracks[i][0]), torch.tensor(tracks[i][3]), torch.tensor(tracks[i][1]), tracks[i][2]) for i in range(args.batch_size)]
            seq_batch_sorted = sorted(seq_dict, key=lambda x: len(x[0]), reverse=True)
            label_batch = [seq_batch_sorted[i][1].float().to(device) for i in range(args.batch_size)]
            seq_batch = [seq_batch_sorted[i][0] for i in range(args.batch_size)]
            seq_lens = [seq_batch[i].shape[0] for i in range(args.batch_size)]
            track_batch = [seq_batch_sorted[i][2].to(device) for i in range(args.batch_size)]

            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_batch, batch_first=True)
            packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).to(device)

            out = model(packed_seq_batch, track_batch)
            loss = loss_function(out, torch.stack(label_batch, 0).view(args.batch_size, 1)).to(device)
            val_loss += loss.data.item()
            y_scores += [out[i].data.item() for i in range(args.batch_size)]
            yval +=  [label_batch[i].data.item() for i in range(args.batch_size)]
            val_eProbHT += [seq_batch_sorted[i][3] for i in range(args.batch_size)]
    
    val_loss /= val_size
    val_loss_list.append(val_loss)
    val_auc = roc_auc_score(yval, y_scores)
    val_eProbHT_auc = roc_auc_score(yval, val_eProbHT)

    if(report): print(f"Epoch {epoch} - Val loss: {np.round(val_loss,3)} - Val AUC: {np.round(val_auc,3)} / eProbHT AUC: {np.round(val_eProbHT_auc,3)}")
    
    step = (epoch + 1) * args.train_size

    log_scalar('Validation_loss', val_loss, step)
    log_scalar('Validation AUC', val_auc, step)
    log_scalar('Validation eProbHT AUC', val_eProbHT_auc, step)


def test(test = test_set, isMC = "MC"):
    model.eval()
    y_scores = []
    ytest = []
    test_eProbHT = []
    test_loss = 0
    with torch.no_grad():
        for tracks in test:
            seq_dict  = [(torch.tensor(tracks[i][0]), torch.tensor(tracks[i][3]), torch.tensor(tracks[i][1]), tracks[i][2]) for i in range(args.batch_size)]
            seq_batch_sorted = sorted(seq_dict, key=lambda x: len(x[0]), reverse=True)
            label_batch = [seq_batch_sorted[i][1].float().to(device) for i in range(args.batch_size)]
            seq_batch = [seq_batch_sorted[i][0] for i in range(args.batch_size)]
            seq_lens = [seq_batch[i].shape[0] for i in range(args.batch_size)]
            track_batch = [seq_batch_sorted[i][2].to(device) for i in range(args.batch_size)]

            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_batch, batch_first=True)
            packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).to(device)

            out = model(packed_seq_batch, track_batch)
            loss = loss_function(out, torch.stack(label_batch, 0).view(args.batch_size, 1)).to(device)
            #loss = np.sum([loss_function(out[i], label_batch[i].float()) for i in range(args.batch_size)]).to(device)
            test_loss += loss.data.item()
            y_scores += [out[i].data.item() for i in range(args.batch_size)]
            ytest +=  [label_batch[i].data.item() for i in range(args.batch_size)]
            test_eProbHT += [seq_batch_sorted[i][3] for i in range(args.batch_size)]
    
    test_loss /= test_size
    test_auc = roc_auc_score(ytest, y_scores)
    test_eProbHT_auc = roc_auc_score(ytest, test_eProbHT)

    step = (epoch + 1) * args.train_size

    log_scalar(f'{isMC}_Test_loss', test_loss, step)
    log_scalar(f'{isMC}_Test AUC', test_auc, step)
    log_scalar(f'{isMC}_Test eProbHT AUC', test_eProbHT_auc, step)


    print(f"Test sample - Number of muons: {len(ytest) - np.sum(ytest)} - Number of electrons: {np.sum(ytest)}")
    fpr, tpr, thresholds = roc_curve(ytest, y_scores)
    #Find the 90% efficiency threshold and corresponding True Positive Rate (tpr) and False Positive Rate (fpr)
    lstm_tpr = max([k for k in tpr if (k < 0.9 + 0.005 and k > 0.9 - 0.005)])
    lstm_tpr_index = list(tpr).index(lstm_tpr)
    lstm_fpr = fpr[lstm_tpr_index]
    lstm_threshold = thresholds[lstm_tpr_index]
    if(report): print(f"LSTM TPR for {lstm_tpr} - LSTM FPR = {lstm_fpr} - LSTM Threshold = {lstm_threshold}")


    #plot y_scores for electrons and muons seperately and the combined ROC curve
    e_y_scores = [y_scores[k] for k in range(len(y_scores)) if ytest[k] == 1]
    m_y_scores = [y_scores[k] for k in range(len(y_scores)) if ytest[k] == 0]
    
    e_test_eProbHT = [test_eProbHT[k] for k in range(len(test_eProbHT)) if ytest[k] == 1]
    m_test_eProbHT = [test_eProbHT[k] for k in range(len(test_eProbHT)) if ytest[k] == 0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
    n1, bins, patches = ax2.hist(e_y_scores, 50, color = "C0", label = "Electron RNN scores", alpha = 0.5, density = True)
    ax2.hist(e_test_eProbHT, 50, color = "C1", label = "Electron eProbHT scores", alpha = 0.5, density = True)
    ax2.legend()
    ax2.set_xlabel("Scores")

    n2, bins, patches = ax3.hist(m_y_scores, 50, color = "C0", label = "Muon RNN scores", alpha = 0.5, density = True)
    ax3.hist(m_test_eProbHT, 50, color = "C1", label = "Muon eProbHT scores", alpha = 0.5, density = True)
    ax3.legend()
    ax3.set_xlabel("Scores")
    fig.suptitle(f"{isMC} - Score distributions (normalized) and ROC curve")

    print(f"Epoch {args.epochs} - Test loss: {np.round(test_loss,3)} -- Test AUC: {np.round(test_auc,3)} -- eProbHT AUC: {np.round(test_eProbHT_auc,3)}")
        
    fpr, tpr, thresholds = roc_curve(ytest, y_scores)
    ax1.plot(tpr, fpr, linewidth = 2, label= f"RNN ({np.round(test_auc,4)})", color = "C0")
    fpr, tpr, thresholds = roc_curve(ytest, test_eProbHT)
    ax1.plot(tpr, fpr, linewidth = 2, label= f"eProbHT ({np.round(test_eProbHT_auc,4)})", color = "C1")
    ax1.axvline(x=0.9,linestyle='--',color='k', label = "%90 eff.")
    ax1.legend()
    ax1.set_xlabel("Signal Efficiency")
    ax1.set_ylabel("Background Efficiency")
    plt.savefig(f"plots/{isMC}_{model_name}_epoch{args.epochs}")
    if(show_plots): plt.show()  
    plt.close()

def log_scalar(name, value, step):

    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)

with mlflow.start_run():
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param("track_features", track_features)
    mlflow.log_param("hit_features", hit_features)
    mlflow.log_param('comment', comment)
    

    # Perform the training
    if(isTrain):
        for epoch in range(1, args.epochs + 1):
            print("Training...")
            train(epoch)
            print("Validating")
            validate(epoch)

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
            os.path.join(mlflow.get_artifact_uri(), "events"))
    epoch = args.epochs
    
    print("MC Testing...")
    test(test = test_set, isMC = "MC")

    print("DATA Testing...")
    test(test = test_set_data, isMC = "DATA")



