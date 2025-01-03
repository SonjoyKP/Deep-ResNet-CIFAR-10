import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from NetWork import ResNet, resnet18
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = resnet18(use_residual=config.use_residual, use_bn=config.use_bn)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(),  self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1/1.5)
        self.train_loss_history = []
        self.testing_loss_history = []
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Manually update or use scheduler from pytorch
            
            ### YOUR CODE HERE
            epoch_loss = 0.0  # Track loss for the epoch
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                start_idx = i * self.config.batch_size
                end_idx = (i + 1) * self.config.batch_size
                batch_x = np.array([parse_record(record, training=True) for record in curr_x_train[start_idx:end_idx]])
                batch_y = curr_y_train[start_idx:end_idx]

                # Convert to torch tensors and move to configured device
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(self.config.device)
                batch_y = torch.tensor(batch_y, dtype=torch.long).to(self.config.device)

                # Forward pass
                logits = self.network(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            

            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            self.train_loss_history.append(avg_loss)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds, LR: {:.6f}.'.format(epoch, avg_loss, duration, self.optimizer.param_groups[0]['lr']))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)
            
            self.scheduler.step()  # Update learning rate per epoch
                
        ### YOUR CODE HERE
        #Plot the curves
        # Plot the training loss curve
        plt.plot(self.train_loss_history, label=f"Training loss - LR: {self.config.lr}, Res: {self.config.use_residual}, BN: {self.config.use_bn}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()

        ### YOUR CODE HERE


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            with torch.no_grad():
                for i in tqdm(range(x.shape[0])):
                    ### YOUR CODE HERE
                    # Preprocess the input record and make predictions
                    record = parse_record(x[i], training=False)
                    record = torch.tensor(record, dtype=torch.float32).unsqueeze(0).to(self.config.device)

                    logits = self.network(record)
                    pred = torch.argmax(logits, dim=1).item()
                    preds.append(pred)
                    ### END CODE HERE

            y_tensor = torch.tensor(y, dtype=torch.long).to(self.config.device)
            preds_tensor = torch.tensor(preds, dtype=torch.long).to(self.config.device)
            accuracy = torch.sum(preds_tensor == y_tensor).item() / y_tensor.size(0)
            print('Checkpoint {} Test accuracy: {:.4f}'.format(checkpoint_num, accuracy))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location=self.config.device)
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def train_test_accuracy(self, x_train, y_train, x_test, y_test, max_epoch, save_checkpoint_model = False):
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            self.network.train()

            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Manually update or use scheduler from pytorch
            
            ### YOUR CODE HERE
            epoch_loss = 0.0  # Track loss for the epoch
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                start_idx = i * self.config.batch_size
                end_idx = (i + 1) * self.config.batch_size
                batch_x = np.array([parse_record(record, training=True) for record in curr_x_train[start_idx:end_idx]])
                batch_y = curr_y_train[start_idx:end_idx]

                # Convert to torch tensors and move to configured device
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(self.config.device)
                batch_y = torch.tensor(batch_y, dtype=torch.long).to(self.config.device)

                # Forward pass
                logits = self.network(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            

            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            self.train_loss_history.append(avg_loss)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds, LR: {:.6f}.'.format(epoch, avg_loss, duration, self.optimizer.param_groups[0]['lr']))

            if save_checkpoint_model and (epoch % self.config.save_interval == 0):
                self.save(epoch)

            # Testing loss every 10 epochs
            if epoch % 10 == 0:
                self.network.eval()
                with torch.no_grad():
                    test_loss = 0
                    for i in range(0, len(x_test), self.config.batch_size):
                        batch_x = np.array([parse_record(record, training=False) for record in x_test[i:i + self.config.batch_size]])
                        batch_y = y_test[i:i + self.config.batch_size]

                        batch_x = torch.tensor(batch_x, dtype=torch.float32).to(self.config.device)
                        batch_y = torch.tensor(batch_y, dtype=torch.long).to(self.config.device)
                        logits = self.network(batch_x)
                        loss = self.loss_fn(logits, batch_y)
                        test_loss += loss.item()
                    avg_testing_loss = test_loss / (len(x_test) // self.config.batch_size)
                    self.testing_loss_history.append(avg_testing_loss)
                    print(f"Epoch {epoch}/{max_epoch}, Testing Loss: {avg_testing_loss:.4f}")
            
            self.scheduler.step()  # Update learning rate per epoch
            
        # Final testing accuracy
        self.network.eval()
        preds = []
        with torch.no_grad():
            for i in tqdm(range(x_test.shape[0])):
                record = parse_record(x_test[i], training=False)
                record = torch.tensor(record, dtype=torch.float32).unsqueeze(0).to(self.config.device)

                logits = self.network(record)
                pred = torch.argmax(logits, dim=1).item()
                preds.append(pred)

        y_tensor = torch.tensor(y_test, dtype=torch.long).to(self.config.device)
        preds_tensor = torch.tensor(preds, dtype=torch.long).to(self.config.device)
        final_accuracy = torch.sum(preds_tensor == y_tensor).item() / y_tensor.size(0)

        print(f"\nFinal Testing Accuracy: {final_accuracy:.4f}")
        
        # Plot training and testing loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_epoch + 1), self.train_loss_history, label="Training Loss")
        plt.plot(range(10, max_epoch + 1, 10), self.testing_loss_history, label="Testing Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves - LR:{self.config.lr}, Res: {self.config.use_residual}, BN: {self.config.use_bn}, Acc: {final_accuracy:.4f}")
        plt.legend()
        plt.show()

