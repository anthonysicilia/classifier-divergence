import torch
from tqdm import tqdm

class NoSchedule:

    def __init__(self, *args, **kwargs):
        pass
    
    def step(*args, **kwargs):
        pass

class Mean:

    def __init__(self):
        self.values = []
        self.counts = []
    
    def update(self, value, weight=1):
        self.values.append(value)
        self.counts.append(weight)
    
    def compute(self):
        if len(self.counts) > 0:
            return sum(self.values) / sum(self.counts)
        else:
            raise ArithmeticError(
                'Tried to compute average of empty array!')

def to_device(iterator, device):
    for x in iterator:
        arr = []
        for xi in x:
            if type(xi) == dict:
                xi = {k : v.to(device) for k,v in xi.items()}
                arr.append(xi)
            else:
                arr.append(xi.to(device))
        yield tuple(arr)

def lazy_kwarg_init(init, **kwargs):

    class LazyCallable:

        def __init__(self, init, kwargs):
            self.init = init
            self.kwargs = kwargs
        
        def __call__(self, *args):
            return self.init(*args, **kwargs)
    
    return LazyCallable(init, kwargs)

class Commands:

    ERR_MSG = 'class Commands is abstract. Please implement a ' \
        'custom subclass based on the provided prototype. ' \
        'Otherwise, please use the pre-implemented ' \
        'commands (class Default).'

    def __init__(self):
        raise NotImplementedError(self.ERR_MSG)
    
    def extract_x(self, data):
        """
        Exracts unlabeled features from a batch.

        Parameters
        ----------
        data : either a batch returned by torch.utils.data.DataLoader
            or an individual data point returned 
            by torch.utils.data.Dataset

        Returns
        -------
        torch.tensor with unlabeled (batch of) features

        Examples
        --------
        (data as a batch)
        >>> # Code for retrieving predictions of a model & adding to array
        >>> dataset = Dataset() # some PyTorch Dataset
        >>> dataloader = torch.utils.DataLoader(Dataset(), batch_size=64)
        >>> model_preds = []
        >>> commands = Commands()
        >>> for data in dataloader:
        >>>     x = commands.extract_x(data)
        >>>     yhat = commands.score(x, model)
        >>>     yhat = yhat.argmax(dim=1)
        >>>     for i in range(yhat.size(0)):
        >>>         model_preds.append(yhat[i].cpu().item())

        (individual data)
        >>> # Code for retrieving an indivudal data point
        >>> dataset = Dataset() # some PyTorch Dataset
        >>> x = commands.extract_x(dataset.__getitem__(index))
        """
        raise NotImplementedError(self.ERR_MSG)
    
    def score(self, x, model):
        """
        Assigns scores for each label by applying the provided 
        classifier to the batch of unlabeled features.

        Parameters
        ----------
        x : unlabeled (batch of) feature tensors
        model : classifier used to assign scores

        Returns
        -------
        a tensor of scores with shape (batch size, # classes)

        Examples
        --------
        >>> dataset = Dataset() # some PyTorch Dataset
        >>> dataloader = torch.utils.DataLoader(Dataset(), batch_size=64)
        >>> model_preds = []
        >>> commands = Commands()
        >>> for batch in dataloader:
        >>>     x = commands.extract_x(data)
        >>>     yhat = commands.score(x, model)
        >>>     yhat = yhat.argmax(dim=1)
        >>>     for i in range(yhat.size(0)):
        >>>         model_preds.append(yhat[i].cpu().item())
        """
        raise NotImplementedError(self.ERR_MSG)

class Default(Commands):

    def __init__(self):
        pass

    def extract_x(self, data):
        """
        Assumes data is a tuple and returns the first entry.

        See parent class Commands for more details.
        """
        return data[0]
    
    def score(self, x, model):
        """
        Assumes scores are directly output by the model
        as below:
        >>> scores = model(x)

        See parent class Commands for more details.
        """
        return model(x)

class DisagreementSet(torch.utils.data.Dataset):

    def __init__(self, a, b, model, dataloader, commands, device='cpu'):
        super().__init__()
        self.a = a
        self.b = b
        self.commands = commands
        self.weight_a = (max(len(a), len(b))) / len(a)
        self.weight_b = (max(len(a), len(b))) / len(b)
        self.predictions = []
        self.labels = []
        model.eval()
        with torch.no_grad():
            # predictions for a
            dataset = to_device(dataloader(self.a), device)
            for data in dataset:
                x = self.commands.extract_x(data)
                yhat = self.commands.score(x, model)
                yhat = yhat.argmax(dim=1)
                for i in range(yhat.size(0)):
                    self.labels.append(yhat[i].cpu().item())
                    self.predictions.append(yhat[i].cpu().item())
            # predictions for b
            dataset = to_device(dataloader(self.b), device)
            for data in dataset:
                x = self.commands.extract_x(data)
                yhat = self.commands.score(x, model)
                # use the second prediction, could also randomly
                # sample, or pick lowest. Reasoning is, it should 
                # be easiest to confuse the first and second 
                # most confident prediction
                faux_yhat = yhat.topk(k=2, dim=1).indices[:, 1]
                yhat = yhat.argmax(dim=1)
                for i in range(yhat.size(0)):
                    self.labels.append(faux_yhat[i].cpu().item())
                    self.predictions.append(yhat[i].cpu().item())
        
    def __len__(self):
        return len(self.a) + len(self.b)
    
    def __getitem__(self, index):
        pred = self.predictions[index]
        label = self.labels[index]
        if index >= len(self.a):
            index = index - len(self.a)
            x = self.commands.extract_x(self.b.__getitem__(index))
            return (x, label, self.weight_b, pred, 1)
        else:
            x = self.commands.extract_x(self.a.__getitem__(index))
            return (x, label, self.weight_a, pred, 0)

class Discrepancy:

    def __init__(self, model_init, 
        model_init_params={}, batch_size=64,
        epochs=100, optim_class=torch.optim.SGD, 
        optim_params={'lr' : 1e-2, 'momentum' : 0.9},
        scheduler_class=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size' : 75, 'gamma' : 0.1},
        pass_loss_to_sched=False, 
        commands=Default, commands_params={},
        device='cpu', verbose=False):
        """
        A class for computing the h-discrepency,
        a.k.a., h\Delta\mathcal{H}-divergence.
        Computing the statistic requires training a new model,
        so parameters defined at initialization are primarily
        related to model training. Typically, you should 
        choose these to be identical to the training parameters
        used when training the model you intend to use for inference.

        Parameters
        ----------
        model_init : a callable used to initialize a new model.
        model_init_params : a dictionary of parameters to pass to model_init,
            for example, as done in the following line
            >>> new_model = model_init(**model_init_params)
        optim_class : a class (callable) implemented in torch.optim, to be 
            used for training a new model.
        optim_params : a dictionary of parameters to pass to optim_class,
            for example, as done in the following lines
            >>> new_model = model_init(**model_init_params)
            >>> optim = optim_class(new_model.parameters(), **optim_params)
        scheduler_class : a class (callable) implemented in 
            torch.optim.lr_scheduler, to be used for training a new model.
        scheduler_params : a dictionary of parameters to pass to 
            scheduler_class, for example, as used in the following lines
            >>> new_model = model_init(**model_init_params)
            >>> optim = optim_class(new_model.parameters(), **optim_params)
            >>> sched = scheduler_class(optim, **scheduler_params)
        pass_loss_to_sched : when true, pass the loss to scheduler on each epoch step,
            for example, when using torch.optim.lr_scheduler.ReduceLROnPlateau
        commands : an implementation of class Commands (callable).
        device : a torch device on which to do computation.
        verbose : if true, give updates on how computation is progressing.
        """
        self.model_init = lambda: model_init(**model_init_params)
        self.optim_init = lambda p: optim_class(p, **optim_params)
        self.sched_init = lambda o: scheduler_class(o, **scheduler_params)
        self.epochs = epochs
        self.pass_loss_to_sched = pass_loss_to_sched
        self.commands = commands(**commands_params)
        loader = torch.utils.data.DataLoader
        self.train_dataloader = lazy_kwarg_init(loader,
            batch_size=batch_size, 
            shuffle=True)
        self.test_dataloader = lazy_kwarg_init(loader,
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False)
        self.device = device
        self.verbose = verbose
    
    def compute(self, h, a, b):
        """
        Compute the h-discrepancy given two datasets and a model.

        Parameters
        ----------
        h : an instance of torch.nn.Module, typically pre-trained on 
            some source data. This corresponds to the model you would 
            like to use for inference on some target data.
        a : torch.utils.Dataset (e.g., the source data).
        b : torch.utils.Dataset (e.g., the target data).

        Returns
        -------
        the h-discrepancy between dataset a and dataset b
        """
        if self.verbose:
            print('Computation Step 1/2...')
        x = self._asymmetric_compute(h, a, b)
        if self.verbose:
            print('Computation Step 2/2...')
        y = self._asymmetric_compute(h, b, a)
        return max(x, y)
    
    def _asymmetric_compute(self, h, a, b):
        dataset = DisagreementSet(a, b, h.to(self.device), 
            self.test_dataloader, self.commands, device=self.device)

        # training phase
        iterator = to_device(self.train_dataloader(dataset), self.device)
        model = self.model_init().to(self.device)
        optim = self.optim_init(model.parameters())
        sched = self.sched_init(optim)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        if self.verbose:
            epoch_iterator = tqdm(range(self.epochs))
        else:
            epoch_iterator = range(self.epochs)

        for _ in epoch_iterator:
            avg_loss = Mean()
            for x, y, w, *_ in iterator:
                optim.zero_grad()
                yhat = self.commands.score(x, model)
                l = loss_fn(yhat, y)
                loss = (l * w).sum() / w.sum()
                loss.backward()
                optim.step()
                avg_loss.update(loss.item())
            if self.pass_loss_to_sched:
                sched.step(avg_loss)
            else:
                sched.step()
            if self.verbose:
                epoch_iterator.set_postfix({'loss' : avg_loss.compute()})
        
        # testing phase
        iterator = to_device(self.test_dataloader(dataset), self.device)
        prob_dis_a = Mean()
        prob_dis_b = Mean()
        with torch.no_grad():
            model.eval()
            for x, _, _, h_pred, z in iterator:
                yhat = model(x).argmax(dim=1)
                a_ind = (yhat[z == 0] != h_pred[z == 0])
                b_ind = (yhat[z == 1] != h_pred[z == 1])
                prob_dis_a.update(a_ind.sum().item(), 
                    weight=len(a_ind))
                prob_dis_b.update(b_ind.sum().item(), 
                    weight=len(b_ind))

        return abs(prob_dis_a.compute() - prob_dis_b.compute())