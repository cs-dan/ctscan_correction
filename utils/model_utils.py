### Dependencies

# AI processes
import torch
import torch.nn as nn
from torchvision import transforms

# Plotting
import matplotlib.pyplot as pyp


### Configs & Variables

transform_config = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


### Functions

def load_model(model):
    """
    Loads the AI model.
    """
    print('here')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def plot_stats(loss, accuracy, num_epoch):
    """
    Plots loss and accuracy over training session.
    """

    # Arguments
    run_loss = loss
    run_accuracy = accuracy
    num_iters = range(1, num_epoch + 1)

    # Prep
    graphs = [run_loss, run_accuracy]

    # Figure, Axis creation
    figure, (axis_1, axis_2) = pyp.subplots(1, 2, figsize=(24, 8))

    # Ploting using pyplot
    axis_1.plot(num_iters, graphs[0], label= "Output", color= 'teal', linewidth= 2)
    axis_1.set_title("Loss by Epoch", fontsize= 16, fontweight= "bold")
    axis_1.set_xlabel("Epoch", fontsize= 12)
    axis_1.set_ylabel("Loss", fontsize= 12)
    axis_1.grid(True, which= "both", linestyle= "-", linewidth= 1)
    axis_1.set_facecolor("#f0f0f0")
    axis_1.legend(loc= "best")

    axis_2.plot(num_iters, graphs[1], label= "Loss", color= 'violet', linewidth= 2)
    axis_2.set_title("Accuracy by Epoch", fontsize= 16, fontweight= "bold")
    axis_2.set_xlabel("Epoch", fontsize= 12)
    axis_2.set_ylabel("Accuracy", fontsize= 12)
    axis_2.grid(True, which= "both", linestyle= "-", linewidth= 0.5)
    axis_2.set_facecolor("#f0f0f0")
    axis_2.legend(loc= "best")

    pyp.tight_layout()

    # Save and show
    pyp.savefig('training_run_graphs.png', dpi= 300)
    pyp.show()

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom data class for loading data
    """
    def __init__(self, inputs, labels):
        self.inputs= inputs
        self.labels= labels
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6) #6
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12) #12
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18) #18

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        
        # Input shape
        size = x.shape[2:]

        # Convolution block
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3 = self.relu(self.conv3(x))
        conv4 = self.relu(self.conv4(x))

        # Global pooling
        global_pool = self.global_avg_pool(x)
        global_pool = nn.functional.interpolate(global_pool, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        x = torch.cat([conv1, conv2, conv3, conv4, global_pool], dim=1)
        x = self.out_conv(x)

        # Output
        return x

class Tiny_Classifier_A(nn.Module):
    """ Small classifier w/ ASPP """

    def __init__(self, num_classes):
        super(Tiny_Classifier_A, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 96, kernel_size= 5, stride= 1, dilation= 2) #1, 96
        self.conv2 = nn.Conv2d(96, 64, kernel_size= 3, stride= 2) #96, 64

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set ASPP module
        self.aspp = ASPP(in_channels= 64, out_channels= 32) #64, 32

        # Set fully connected layers
        self.fc1 = nn.Linear(32, num_classes) #32, num_classes

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # Initial convolutional blocks
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))

        # ASPP block
        x = self.aspp(x)

        # Average pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)

        # Pass to fc layer 
        x = self.fc1(x)

        # Output
        return x

class Scan_Classifier(nn.Module):
    """
    The one to use.
    """

    def __init__(self, num_classes):
        super(Scan_Classifier, self).__init__()

        # Setting convolution layers
        self.conv1 = nn.Conv2d(1, 60, kernel_size= 5, stride= 1)
        self.conv2 = nn.Conv2d(60, 120, kernel_size= 5, stride= 1)
        self.conv3 = nn.Conv2d(120, 60, kernel_size= 3, stride= 2)

        # Fully connected part
        self.fc = nn.Linear(60, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier(nn.Module):
    """
    Very small classifier
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 60, kernel_size= 5, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(60, 120, kernel_size= 5, stride= 1, dilation= 2)
        self.conv3 = nn.Conv2d(120, 60, kernel_size= 5, stride= 1)

        # Set fully connected layers
        self.fc1 = nn.Linear(60, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_DM(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_DM, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 120, kernel_size= 3, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(120, 80, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(80, 40, kernel_size= 3, stride= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(40, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_XM(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_XM, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 160, kernel_size= 5, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(160, 120, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(120, 80, kernel_size= 3, stride= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(80, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_SM(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_SM, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 120, kernel_size= 5, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(120, 80, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(80, 40, kernel_size= 5, stride= 2, dilation= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(40, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_SPF(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_SPF, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 48, kernel_size= 3, stride= 1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size= 5, stride= 2, dilation= 2)
        self.conv3 = nn.Conv2d(96, 48, kernel_size= 3, stride= 2, dilation= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(48, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        # x = self.maxpool(x)
        # print(x.size())

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_SP(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_SP, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 48, kernel_size= 3, stride= 1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size= 3, stride= 2, dilation= 2)
        self.conv3 = nn.Conv2d(96, 48, kernel_size= 3, stride= 2, dilation= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(48, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_SP2(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_SP2, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 38, kernel_size= 5, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(38, 76, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(76, 38, kernel_size= 5, stride= 2, dilation= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(38, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        # x = self.maxpool(x)
        # print(x.size())

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_D(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_D, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 48, kernel_size= 3, stride= 1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size= 5, stride= 2, dilation= 2)
        self.conv3 = nn.Conv2d(96, 48, kernel_size= 3, stride= 2, dilation= 2)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(48, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

class Tiny_Classifier_F(nn.Module):
    """
    Very small classifier alt
    """

    def __init__(self, num_classes):
        super(Tiny_Classifier_F, self).__init__()

        # Set convolution layers
        self.conv1 = nn.Conv2d(1, 40, kernel_size= 5, stride= 1, dilation= 2)
        self.conv2 = nn.Conv2d(40, 80, kernel_size= 3, stride= 1, dilation= 2)
        self.conv3 = nn.Conv2d(80, 40, kernel_size= 3, stride= 1)

        # Set pooling
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Set fully connected layers
        self.fc1 = nn.Linear(40, num_classes)

        # Set misc ops
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):

        # Pass through convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))

        # Average pooling and flattening
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim= 1)
        
        # Pass to fc layer
        x = self.fc1(x)

        return x

def training_loop(model, dataloader, loss_fn, optimizer, scheduler, num_epochs, device):
    """
    Generic training loop for the classifier
    """
    model.train()

    # Stopping criteria
    tolerance = 1e-2
    patience_count = 8
    
    # Tracking variables
    previous_loss = float('inf')
    no_improvement_count = 0

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_total, epoch_correct = 0, 0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            
            labels = labels.unsqueeze(1).float()

            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, idxs = torch.max(outputs, 1)
            epoch_total += labels.size(0)
            epoch_correct += (idxs == labels).sum().item()
            if inputs.shape[0] >= 8: print('batch done')
        
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = (epoch_correct / epoch_total)
        print(f'Epoch {epoch+1} of {num_epochs}\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}')

        # Breaking conditions

        # If near perfect on training accuracy
        if epoch_accuracy >= 0.99:
            break

        # Update stopping condition
        if abs(previous_loss - loss.item()) <= tolerance:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        # Check stopping condition
        if patience_count <= no_improvement_count:
            print(f"Early stopping at epoch {epoch + 1} due to minial improvement.")
            break

        previous_loss = loss.item()

def validation_loop(model, dataloader, device):
    """
    Generic evaluation loop for the model
    """
    model.eval()
    running_loss = 0.0
    num_total, num_correct = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            max_val, idxs = torch.max(outputs, 1)
            num_total += labels.size(0)
            num_correct += (idxs == labels).sum().item()
    
    valid_accuracy = (num_correct / num_total) * 100
    print(f'Accuracy of validation set: {valid_accuracy:.2f}%')

def training_loop_2(model, dataloader, loss_fn, optimizer, scheduler, num_epochs, device):
    """
    Generic training loop for the classifier
    """
    model.train()

    # Stopping criteria
    tolerance = 1e-2
    patience_count = 8
    
    # Tracking variables
    previous_loss = float('inf')
    no_improvement_count = 0

    # Stat collections
    run_loss = []
    run_accuracy = []

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_total, epoch_correct = 0, 0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            
            labels = labels.unsqueeze(1).float()

            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            out = torch.sigmoid(outputs) > 0.5
            epoch_total += labels.size(0)
            epoch_correct += (out == labels).sum().item()
            # if inputs.shape[0] >= 8: print('batch done')
        
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = (epoch_correct / epoch_total)
        print(f'Epoch {epoch+1} of {num_epochs}\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}')

        # Stat collection
        run_loss.append(epoch_loss)
        run_accuracy.append(epoch_accuracy)

        # Breaking conditions

        # If near perfect on training accuracy
        if epoch_accuracy >= 0.99:
            break

        # Update stopping condition
        if abs(previous_loss - loss.item()) <= tolerance:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        # Check stopping condition
        if patience_count <= no_improvement_count:
            print(f"Early stopping at epoch {epoch + 1} due to minial improvement.")
            break

        previous_loss = loss.item()
    
    plot_stats(run_loss, run_accuracy, num_epochs)

def validation_loop_2(model, dataloader, device):
    """
    Generic evaluation loop for the model
    """
    model.eval()
    num_total, num_correct = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            labels = labels.unsqueeze(1).float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            out = torch.sigmoid(outputs) > 0.5
            num_total += labels.size(0)
            num_correct += (out == labels).sum().item()
    
    valid_accuracy = (num_correct / num_total) * 100
    print(f'Accuracy of validation set: {valid_accuracy:.2f}%')
