# scPairing documentation

```
class scPairing.scPairing(adata1, adata2, mod1_type='rna', mod2_type='atac', batch_col=None, transformed_obsm='X_pca', counts_layer=None, use_decoder=False, emb_dim=10, encoder_hidden_dims=(128,), decoder_hidden_dims=(128,), reconstruct_mod1_fn=None, reconstruct_mod2_fn=None, seed=None, **model_kwargs)
```

Parameters
----------
* ``adata1`` (``AnnData``)  
    AnnData object corresponding to the first modality of a multimodal single-cell dataset
* ``adata2`` (``AnnData``)  
    AnnData object corresponding to the second modality of a multimodal single-cell dataset
* ``mod1_type`` (``Literal['rna', 'atac', 'protein', 'other']``)  
    The modality type of adata1. One of the following:

    * ``'rna'`` - for scRNA-seq data modeled with a negative binomial distribution
    * ``'atac'`` - for scATAC-seq data modeled with a Bernoulli distribution
    * ``'protein'`` for epitope data modeled with a negative binomial distribution
    * ``'other'`` for other data modalities modeled with a Gaussian distribution
* ``mod2_type`` (``Literal['rna', 'atac', 'protein', 'other']``)  
    The modality type of adata2. The options are identical to mod1_type.
* ``batch_col`` (``Optional[str]``)  
    Column in ``adata1.obs`` and ``adata2.obs`` corresponding to batch information
* ``transformed_obsm`` (``Union[str, List[str]]``)  
    Key(s) in ``adata1.obsm`` and ``adata2.obsm`` corresponding to the low-dimension
    representations of each individual modality. If a string is provided, the same key will
    be applied to both ``adata1.obsm`` and ``adata2.obsm``. If ``None`` is provided,
    the representations will be taken from ``adata1.X`` and/or ``adata2.X``.
* ``counts_layer`` (``Optional[Union[str, List[str]]]``)  
    Key(s) in ``adata1.layers`` and ``adata2.layers`` corresponding to the raw counts for each modality.
    If a string is provided, the same key will be applied to both ``adata1.layers`` and ``adata2.layers``.
    If ``None`` is provided, raw counts will be taken from ``adata1.X`` and/or ``adata2.X``.
* ``use_decoder`` (``bool``)  
    Whether to train a decoder to reconstruct the counts on top of the low-dimension representations.
* ``emb_dim`` (``int``)  
    Dimension of the hyperspherical latent space
* ``encoder_hidden_dims``  (``Sequence[int]``)  
    Number of nodes and depth of the encoder
* ``decoder_hidden_dims``  (``Sequence[int]``)  
    Number of nodes and depth of the decoder
* ``reconstruct_mod1_fn`` (``Optional[Callable]``)  
    Custom function that reconstructs the counts from the reconstructed low-dimension representations
    for the first data modality.
* ``reconstruct_mod2_fn`` (``Optional[Callable]``)  
    Custom function that reconstructs the counts from the reconstructed low-dimension representations
    for the second modality.
* ``seed`` (``Optional[int]``) 
    Random seed for model reproducibility.
* ``**model_kwargs``  
    Keyword args for scPairing.

<br/><br/>

```
scPairing.train(epochs, batch_size=2000, restart_training=False)
```
Train the model.
        
Parameters
----------
* ``epochs`` (``int``)  
    Number of epochs to train the model.
* ``batch_size`` (``int``)  
    Minibatch size. Larger batch sizes recommended in contrastive learning.
* ``restart_training`` (``bool``)  
    Whether to re-initialize model parameters and train from scratch, or
    to continue training from the current parameters.
* ``**trainer_kwargs``  
    Keyword arguments for UnsupervisedTrainer

<br/><br/>
```
scPairing.get_latent_representation(adata1=None, adata2=None, batch_size=2000)
```
Returns the embeddings for both modalities.

Parameters
----------
* ``adata1`` (``Optional[AnnData]``)  
    AnnData object corresponding to one modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``adata2`` (``Optional[AnnData]``)  
    AnnData object corresponding to the other modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``batch_size`` (``int``)  
    Minibatch size.

<br/><br/>
```
scPairing.get_normalized_expression(adata1=None, adata2=None, batch_size=2000)
```
Returns the reconstructed counts for both modalities.

Parameters
----------
* ``adata1``  (``Optional[AnnData]``)
    AnnData object corresponding to one modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``adata2`` (``Optional[AnnData]``)  
    AnnData object corresponding to the other modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``batch_size`` (``int``)  
    Minibatch size.

<br/><br/>
```
scPairing.get_likelihoods(adata1=None, adata2=None, batch_size=2000)
```
Return the likelihoods for both modalities.

Parameters
----------
* ``adata1`` (``Optional[AnnData]``)  
    AnnData object corresponding to one modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``adata2`` (``Optional[AnnData]``)  
    AnnData object corresponding to the other modality of a multimodal single-cell dataset.
    If not provided, the AnnData provided on initialization will be used.
* ``batch_size`` (``int``)  
    Minibatch size.

<br/><br/>
```
scPairing.get_cross_modality_expression(mod1_to_mod2, adata=None, batch_size=2000)
```
Predict the expression of the other modality given the representation of one modality.

Parameters
----------
* ``mod1_to_mod2`` (``bool``)  
    Whether to compute the cross modality prediction from the first modality
    to the second modality.
* ``adata`` (``Optional[AnnData]``)  
    AnnData object corresponding to one of the modalities in a multimodal
    dataset. If adata is None and ``mod1_to_mod2`` is ``True``, then ``adata1``
    will be used. If adata is None and ``mod1_to_mod2`` is ``False``, then
    ``adata2`` will be used.
* ``batch_size`` (``int``)  
    Minibatch size.

<br/><br/>
```
scPairing.save(dir_path, prefix='', save_optimizer=False, overwrite=False)
```
Save the model.

Parameters
----------
* ``dir_path`` (``str``)  
    Directory to save the model to.
* ``prefix`` (``str``)  
    Prefix to prepend to file names.
* ``save_optimizer`` (``bool``)  
    Whether to save the optimizer.
* ``overwrite`` (``bool``)  
    Whether to overwrite existing directory.

<br/><br/>
```
scPairing.load(dir_path, prefix='')
```
Load an existing model

Parameters
----------
* ``dir_path`` (``str``)  
    Directory to load the model from.
* ``prefix`` (``str``)  
    Prefix to prepend to file names.
