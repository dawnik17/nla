.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style

.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity
   :alt: Maintenance

.. image:: https://img.shields.io/badge/python-3.above-blue.svg
   :target: https://img.shields.io/badge/python-3.above-blue.svg
   :alt: versions


Natural Language Augmentor (NLA)
=================================
As Emmitt Smith would say, *"All models are created equal. Some work harder in training."*

Augmentation, a critical step in the model training pipeline, is a way to get the model
ready for the hardships of the real world. It distorts the smooth training track to
simulate the rocky road. Thus, the more realistic the augmentation, the more realistic
the training data and therefore, sturdier the model.

NLA, Natural Language Augmentor, is the ultimate real and fast way to augment
textual data for use in any model training. It has all the means to create natural
and random textual errors.

The *homophones* module returns augmented homophones for any given word.
The *Augment* module introduce lightning-fast and parallelized streams
of organised (keyboard sensitive) and random errors. The *edge n-gram* and *word boundary*
modules introduce n-gram deletions and word boundary errors respectively. The entire module 
is written in Pytorch and supports both GPUs and CPUs.

NLA is a one-stop-shop for all lexical augmentations of any text.

Installation
============
Use the package manager pip to install *nla*.
Replace 'branch-name' with the name of the branch you want to pull.

.. code-block:: sh

    pip install git+https://github.com/dawnik17/nla.git@branch-name


Usage
=====

There are some arguments that the below functions take
which might need some explanation.

**degree**

- This is an int/float argument which refers to the intensity of
  the augmentation.

**count**

- This is the tentative number of outputs you want per query.

**method**

- *method*, is the method of augmentation. It takes five values,
  'swap', 'insert', 'replace', 'delete' and 'random'.

**device**

- NLA runs on CPU as well as GPU! If you are on a CUDA supported device, NLA will read the GPUs automatically.


For all augmentations
========================

* First build the factory object for your data
* Then you can reuse it for all augmentations
* The data will be shared among all instansiated augmentations
* All augmentations have word level as well as sentence level operations
* For word level operations, the input can still be a list of sentences (NLA tokenizes these sentences internally for word level usage)
* Switching between these modes is extremely easy; one argument flip and you are there
* Checkout the examples below


.. code-block:: python

    from nla import Factory

    data = ["CROCIN TABLET", "ALEX SYRUP", "NORFLOX"]
    batch_size = 2

    # once loaded, we'll reuse this oject for all augmentations below
    augment = Factory(queries=data, batch_size=batch_size)


Homophones
==========

Generates homophones for words
-------------------------------

.. code-block:: python

    homophone = augment(aug="homophone", mode="word")

    """Get homophones for a list of words

        Args:
            beam_width (int): number of options to consider at each character prediction
            count_per_query (int): result count per query
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            List[tuple]: list of (homophone, query)
    """

    beam_width = 3
    count_per_query = 5 # NOTE - the count_per_query variable isn't very precise; play around with it to get the desired count.

    print(homophone(beam_width=beam_width, count_per_query=count_per_query))

.. code-block:: bash

    >> 100%|██████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.76it/s]
       [
        ('CROCEN', 'CROCIN'), 
        ('CROSIN', 'CROCIN'), 
        ('ALICS', 'ALEX'), 
        ('TABLETTE', 'TABLET'), 
        ('CROCIN', 'CROCIN'), 
        ('SIROP', 'SYRUP'), 
        ('CIROP', 'SYRUP'), 
        ('NARFLOX', 'NORFLOX'), 
        ('NOREFLOCKS', 'NORFLOX'), 
        ('NORFLOX', 'NORFLOX'), 
        ('SYRUP', 'SYRUP'), 
        ('NORFLAX', 'NORFLOX'), 
        ('SIRAP', 'SYRUP'), 
        ('NORPHLOX', 'NORFLOX'), 
        ('SYROP', 'SYRUP'), 
        ('CROSEN', 'CROCIN'), 
        ('OLEX', 'ALEX'), 
        ('ALEXX', 'ALEX'), 
        ('ALEX', 'ALEX'), 
        ('TABLITE', 'TABLET'), 
        ('TABLIT', 'TABLET'), 
        ('TABLATE', 'TABLET'), 
        ('ALLEX', 'ALEX'), 
        ('TABLET', 'TABLET')]


Generates homophones for sentences
----------------------------------

.. code-block:: python

    homophone = augment(aug="homophone", mode="sentence")

    """Get homophones for a list of words

        Args:
            beam_width (int): number of options to consider at each character prediction
            count_per_query (int): result count per query
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            List[tuple]: list of (homophone, query)
    """

    beam_width = 3
    count_per_query = 5 # NOTE - the count_per_query variable isn't very precise; play around with it to get the desired count.

    print(homophone(beam_width=beam_width, count_per_query=count_per_query, ratio=0.8))

.. code-block:: bash

    >> 100%|██████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.76it/s]
       [
        ('NOREFLOCKS ', 'NORFLOX'), 
        ('ALLEX SIRAP', 'ALEX SYRUP'), 
        ('CROCEN TABLETTE', 'CROCIN TABLET'), 
        ('NORFLOX ', 'NORFLOX'), 
        ('CROSEN TABLATE', 'CROCIN TABLET'), 
        ('ALICS SYROP', 'ALEX SYRUP'), 
        ('NARFLOX ', 'NORFLOX'), 
        ('CROSIN TABLITE', 'CROCIN TABLET'), 
        ('ALEX CIROP', 'ALEX SYRUP'), 
        ('CROCIN TABLET', 'CROCIN TABLET'), 
        ('ALICS CIROP', 'ALEX SYRUP'), 
        ('OLEX SYROP', 'ALEX SYRUP')]

Typo
======

* Aug mode "keyboard" mimics the typing errors from a QWERTY keyboard.
* Aug mode "random", is random.
* There are 5 touch modes - "insert", "delete", "replace", "swap", "random".
* For "random" touch mode, diversity of the result set is inversely proportional to the batch size.


Word level typos
-----------------

.. code-block:: python

    typo = augment(aug="typo", mode="word")

    """Introduce typographical errors in a list of words

        Args:
            degree (int): how many max operations do wish to do on one string
            count_per_query (int): result count per query
            touch_mode (str): "insert"/"replace"/"swap"/"delete"/"random"
            aug_mode (str): "keyboard"/"random"

        Returns:
            torch.Tensor: list of typographically augmented words
    """

    print(typo(degree=1, count_per_query=1, touch_mode="insert", aug_mode="keyboard"))
    print(typo(degree=1, count_per_query=1, touch_mode="swap", aug_mode="random"))
    print(typo(degree=1, count_per_query=1, touch_mode="delete", aug_mode="random"))
    print(typo(degree=1, count_per_query=1, touch_mode="replace", aug_mode="keyboard"))


.. code-block:: bash

    >>  100%|██████████| 3/3 [00:00<00:00, 609.28it/s]
        [('CROCOIN', 'CROCIN'), ('NPORFLOX', 'NORFLOX'), ('SYRHUP', 'SYRUP'), ('TABHLET', 'TABLET'), ('ALPEX', 'ALEX')]
        100%|██████████| 3/3 [00:00<00:00, 504.39it/s]
        [('NORLFOX', 'NORFLOX'), ('ATBLET', 'TABLET'), ('SYURP', 'SYRUP'), ('CRCOIN', 'CROCIN'), ('AELX', 'ALEX')]
        100%|██████████| 3/3 [00:00<00:00, 605.44it/s]
        [('NOFLOX', 'NORFLOX'), ('CRCIN', 'CROCIN'), ('ALX', 'ALEX'), ('SYUP', 'SYRUP'), ('TBLET', 'TABLET')]
        100%|██████████| 3/3 [00:00<00:00, 746.23it/s][('ZLEX', 'ALEX'), ('NORCLOX', 'NORFLOX'), ('SGRUP', 'SYRUP'), ('TAGLET', 'TABLET'), ('CRICIN', 'CROCIN')]


Sentence level typos
---------------------

.. code-block:: python

    typo = augment(aug="typo", mode="sentence")

    """Introduce typographical errors in a list of words

        Args:
            degree (int): how many max operations do wish to do on one string
            count_per_query (int): result count per query
            touch_mode (str): "insert"/"replace"/"swap"/"delete"/"random"
            aug_mode (str): "keyboard"/"random"
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            torch.Tensor: list of typographically augmented words
    """

    print(typo(degree=1, count_per_query=1, touch_mode="insert", aug_mode="keyboard", ratio=0.8))
    print(typo(degree=1, count_per_query=1, touch_mode="swap", aug_mode="random", ratio=0.8))
    print(typo(degree=1, count_per_query=1, touch_mode="delete", aug_mode="random", ratio=0.8))
    print(typo(degree=1, count_per_query=1, touch_mode="replace", aug_mode="keyboard", ratio=0.8))


.. code-block:: bash

    >>  100%|██████████| 2/2 [00:00<00:00, 834.02it/s]
        [('NORFLOIX ', 'NORFLOX'), ('ZALEX SYHRUP', 'ALEX SYRUP'), ('CROFCIN TFABLET', 'CROCIN TABLET')]
        100%|██████████| 2/2 [00:00<00:00, 886.18it/s]
        [('RCOCIN TABLTE', 'CROCIN TABLET'), ('AELX SYRPU', 'ALEX SYRUP'), ('NORFOLX ', 'NORFLOX')]
        100%|██████████| 2/2 [00:00<00:00, 860.55it/s]
        [('ORFLOX ', 'NORFLOX'), ('CRCIN ABLET', 'CROCIN TABLET'), ('LEX SYUP', 'ALEX SYRUP')]
        100%|██████████| 2/2 [00:00<00:00, 923.45it/s]
        [('CRICIN TANLET', 'CROCIN TABLET'), ('ALDX SHRUP', 'ALEX SYRUP'), ('NPRFLOX ', 'NORFLOX')]


Edge N-gram
============

* Returns all possible/valid ngrams

Word ngrams
------------


.. code-block:: python

    edgen = augment(aug="ngrams", mode="word")

    """Get all possible edge ngrams for the queries

        Args:
            degree (int): maximum number of characters to be removed per word
            threshold (int): minimum length of a word post edging ngrams

        Returns:
            List[tuple]: list of ngrams
    """

    print(edgen(degree=3, threshold=2, count_per_query=3))


.. code-block:: bash

    >> 100%|█████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.60it/s]
        [
        ('TAB', 'TABLET'), 
        ('CROC', 'CROCIN'), 
        ('SYR', 'SYRUP'), 
        ('TABL', 'TABLET'), 
        ('TABLE', 'TABLET'), 
        ('CRO', 'CROCIN'), 
        ('CROCI', 'CROCIN'), 
        ('NORFL', 'NORFLOX'), 
        ('NORFLO', 'NORFLOX'), 
        ('SYRUP', 'SYRUP'), 
        ('AL', 'ALEX'), 
        ('SYRU', 'SYRUP'), 
        ('SY', 'SYRUP'), 
        ('ALE', 'ALEX'), 
        ('ALEX', 'ALEX'), 
        ('TABLET', 'TABLET'), 
        ('NORF', 'NORFLOX'), 
        ('CROCIN', 'CROCIN')]


Sentence ngrams
----------------

.. code-block:: python

    edgen = augment(aug="ngrams", mode="sentence")

    """Get all possible edge ngrams for the queries

        Args:
            degree (int): maximum number of characters to be removed per word
            threshold (int): minimum length of a word post edging ngrams
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            List[tuple]: list of ngrams
    """

    print(edgen(degree=3, threshold=2, count_per_query=3, ratio=0.8))


.. code-block:: bash

    >> 100%|██████████| 2/2 [00:00<00:00, 465.44it/s]
    [
    ('ALEX SYRUP', 'ALEX SYRUP'), 
    ('CRO TABLE', 'CROCIN TABLET'), 
    ('CROC TABLET', 'CROCIN TABLET'), 
    ('NORFL ', 'NORFLOX'), 
    ('NORF ', 'NORFLOX'), 
    ('ALEX SY', 'ALEX SYRUP'), 
    ('AL SYRUP', 'ALEX SYRUP')]


Word Boundary
===============

Word level word-boundary
-------------------------

.. code-block:: python

    wb = augment(aug="word_boundary", mode="word")

    """Get a list of space errored terms for the queries

        Args:
            degree (int): maximum number of spaces to be introduced per word
            count_per_query (int): result count per query

        Returns:
            List[tuple]: list of space errored words
    """

    print(wb(degree=2, count_per_query=4))


.. code-block:: bash

    >> 100%|█████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.52it/s]
        [
        ('NORF LOX', 'NORFLOX'), 
        ('NOR FLO X', 'NORFLOX'), 
        ('SYRU P', 'SYRUP'), 
        ('S YRU P', 'SYRUP'), 
        ('TABL ET', 'TABLET'), 
        ('N ORF LOX', 'NORFLOX'), 
        ('CRO CI N', 'CROCIN'), 
        ('CROC IN', 'CROCIN'), 
        ('CRO C IN', 'CROCIN'), 
        ('SY R UP', 'SYRUP'), 
        ('CROCI N', 'CROCIN'), 
        ('AL E X', 'ALEX'), 
        ('AL EX', 'ALEX'), 
        ('TABL E T', 'TABLET'), 
        ('NOR FLOX', 'NORFLOX'), 
        ('S YRUP', 'SYRUP'), 
        ('SYR UP', 'SYRUP')]


Sentence level word-boundary
-----------------------------

.. code-block:: python

    wb = augment(aug="word_boundary", mode="sentence")

    """Get a list of space errored terms for the queries

        Args:
            degree (int): maximum number of spaces to be introduced per word
            count_per_query (int): result count per query
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            List[tuple]: list of space errored words
    """

    print(wb(degree=2, count_per_query=4, ratio=0.8))


.. code-block:: bash

    >> 100%|█████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.52it/s]
        [
        ('AL E X SYR UP', 'ALEX SYRUP'), 
        ('C ROCIN T ABL ET', 'CROCIN TABLET'), 
        ('AL E X SYRU P', 'ALEX SYRUP'), 
        ('NORFLO X ', 'NORFLOX'), 
        ('NORF LOX ', 'NORFLOX'), 
        ('NOR FLOX ', 'NORFLOX'), 
        ('AL E X S YRUP', 'ALEX SYRUP'), 
        ('CR OCIN T ABL ET', 'CROCIN TABLET'), 
        ('CR OCIN TABL ET', 'CROCIN TABLET'), 
        ('C ROC IN TABL ET', 'CROCIN TABLET')]


Ensemble 
-----------------------------
Ensemble mode take arguments for each augmentation from config.yml file.
The format of the config file -

1. Keys should be the exact names of augmentations i.e. (typo, ngrams, homophone, word_boundary)

2. To run an assortment of one augmentation with different parameters - 

    1. The keys should be the "augmentation name" followed by a version integer
    2. Example - typo1, typo2, and so on. (refer to the sample config file)

.. code-block:: python

    ensemble = augment(aug="ensemble", mode="ensemble")

    """Get a list of space errored terms for the queries

        Args:
            degree (int): maximum number of spaces to be introduced per word
            count_per_query (int): result count per query
            ratio (int): fraction of words in each sentence we wish to augment

        Returns:
            List[tuple]: list of space errored words
    """

    print(ensemble(count_per_query=5, ratio=0.95, progress_bar=True))


.. code-block:: bash

    >> Typo: swap touch_mode, keyboard aug_mode, count_per_query 5, degree 1: 100%|██████| 3/3 [00:00<00:00, 403.61it/s]
       Typo: random touch_mode, random aug_mode, count_per_query 5, degree 2: 100%|██████| 3/3 [00:00<00:00, 580.96it/s]
       Ngrams: count_per_query 5, threshold 2, degree 3: 100%|███████████████████████████| 3/3 [00:00<00:00, 756.96it/s]
       Word Boundary: count_per_query 5, degree 2: 100%|█████████████████████████████████| 3/3 [00:00<00:00, 380.56it/s]
       Ensemble: : 100%|████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3787.75it/s]
        
        [('RVOCIN TEBLQT', 'CROCIN TABLET'), 
         ('CROCNO TAZLET', 'CROCIN TABLET'), 
         ('COECIN TGZLET', 'CROCIN TABLET'), 
         ('RFOCIN TABLQT', 'CROCIN TABLET'), 
         ('ALE X SY', 'ALEX SYRUP'), 
         ('A LEX SYR', 'ALEX SYRUP'), 
         ('AL E X SYRUP', 'ALEX SYRUP'), 
         ('NORFOKX', 'NORFLOX'), 
         ('NORLVOX', 'NORFLOX'), 
         ('NORFOOX', 'NORFLOX'), 
         ('OMRFLOX', 'NORFLOX'), 
         ('NRKFLOX', 'NORFLOX')]

Note
=====

1. The last method for each augmentation above can be called repeatedly without having to reload the data or the factory class. Once the objects are built - (typo, edgen, wb from above), they can be used any number of times. Like done in the typo example.
2. The precompute data must be a pbz2 file and the path must be as specified -
    
    1. "precompute" folder inside the root nla folder 
    2. The file names should be exactly the name of the augmentation, i.e. [typo.pbz2, ngrams.pbz2, homophone.pbz2, word_boundary.pbz2]


Developer
==========

Every new augmentation must inherit the IAugment class and define a compute method. 
The __call__ method of each augmentation class is the compute method itself.