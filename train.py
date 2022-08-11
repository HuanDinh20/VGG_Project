import torch


def train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch_idx):
    """
    training at one epoch:
    1. Get a batch of data
    2. Zero the optimizer's gradients
    3. Perform inference - that is, gets predictions from model for inputs
    4. Calculate the loss
    5. Calculate backward
    6. tell the optimizer steps
    7. Report loss every 1000 batches
    8. return average loss, to compare with validation loss
    """
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # 1. get inputs and labels from batch
        inputs, labels = data
        # 1.1 move to device
        inputs, labels = inputs.to(device), labels.to(device)

        # 2. zeros the gradients
        optimizer.zero_grad()

        # 3. Perform inference - that is, gets predictions
        outputs = model(inputs)

        # 4. calculate the loss
        loss = loss_fn(outputs, labels)

        # 5. calculate backward
        loss.backward()

        # 6. Tell the optimizer steps
        optimizer.steps()

        # 7.
        running_loss += loss

        # 8.
        if not bool(i % 1000):
            last_loss = running_loss / 1000
            print(f"Batch {i} Loss: {last_loss}")
            tb_x = epoch_idx * len(train_loader) + i + 1
            summary_writer.add_scalar('Loss/Train', last_loss, tb_x)
            running_loss = 0

    return last_loss


def per_epoch_activity(train_loader, test_loader, device, optimizer, model, loss_fn, summary_writer, epochs, timestamp):
    """
    each epoch:
    1. Perform validation by checking
        1.1 get the average loss at train_one_epoch
        1.2 false model train = false
        1.3 calculate validation loss
        1.4 report
    2. save the best model
    """
    best_vloss = 1_000_000
    for epoch in range(epochs):
        # 1.
        model.train(True)
        avg_loss = (train_loader, device, optimizer, model, loss_fn, summary_writer, epoch)
        # 1. 2
        model.train(False)

        running_avg_loss = 0.0
        for i, data in enumerate(test_loader):
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_output = model(val_inputs)

            val_loss = loss_fn(val_output, val_labels)
            running_avg_loss += val_loss

        avg_val_loss = running_avg_loss / (i + 1)
        summary_writer.add_scalars('Training vs. Validation Loss',
                                   {'Training': avg_loss, 'Validation': avg_val_loss},
                                   epoch + 1)

        summary_writer.flush()

        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = fr'saved_model\model_{epoch}_{timestamp}'
            torch.save(model.state_dict(), model_path)

