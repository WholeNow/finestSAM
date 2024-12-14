# finestSAM

Questo progetto è stato realizzato come parte della tesi di laurea presso l'Università degli Studi di Cagliari da:


* [`Marco Pilia`](https://github.com/Marchisceddu)
* [`Simone Dessi`](https://github.com/Druimo)

Lo scopo principale è cercare di effettuare un fine tuning del modello Segment-Anything di MetaAI su un set di dati personalizzati in formato COCO, riuscendo a fornire un'implementazione efficacie per quanto riguarda le predizioni tramite il predittore automatico di SAM.
Il codice sfrutta il framework Fabric di Lightning AI per fornire un implementazione efficiente del modello.

Per leggere l'intera ricerca effettuata per la risoluzione del task è possibile consultare la tesi (in lingua italiana) al [`link`](https://drive.google.com/file/d/1JJwgVJOXWdbUqyN0FSMuvoJxFhoqSF4g/view?usp=sharing)

## Dataset

Inserire un dataset formato coco all'interno della main dir, la cartella dev'essere formattata come segue:

    ```python
    datset/
    ├── images/ # Cartella contenente le immagini del dataset
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── annotations.json # File di annotazioni formato coco
    ```

 Per specificare al modello il nome del dataset inserito modificare le voci correlate come spiegato nella sezione Config [`finestSAM/config.py`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/config.py), di default viene cercata la cartella `"dataset"`.

## Setup

1. Scaricare il checkpoint del modello SAM, la spiegazione è presente in [`finestSAM/sav/`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/sav/)

2. Installare le dipendenze necessarie:

    * Installare tramite pip il file [`requirements.txt`](https://github.com/Marchisceddu/finestSAM/requirements.txt)

          pip install -r requirements.txt

    * Creare un ambiente conda tramite il file [`environment.yaml`](https://github.com/Marchisceddu/finestSAM/environment.yaml)

          conda env create -f environment.yml

## Config

Gli iperparametri necessari al modello sono specificati nel file [`finestSAM/config.py`](https://github.com/Marchisceddu/finestSAM/blob/main/hape_SAM/config.py)

<details>

<summary> Struttura: </summary>
<br>

Generali:
```python
"device": str = "auto" or "gpu" or "cpu", # Hardware su cui eseguire il modello (non è supportata mps, se si usa un mac m1 impostare su cpu)
"num_devices": int # Numero di dispositivi da utilizzare
            or (list str) # Definire quali GPU utilizzare
            or str = "auto", # Sceglie in automatico il dispositivo migliore
"num_nodes": int, # Numero di nodi GPU per l'addestramento distribuito
"seed_device": int / None per random,
"sav_dir": str, # Cartella di output per i salvataggi del modello
"out_dir": str, # Cartella di output per le predizioni

"model": {
    "type": str = "vit_h" or "vit_l" or "vit_b",
    "checkpoint": str, # Nome checkpoint, formato -> nome.pth
},
```

Train:
```python
"seed_dataloader": int / None per random,
"batch_size": int, # Grandezza batch delle immagini
"num_workers": int, # Quanti sottoprocessi utilizzare per il caricamento dei dati (0 -> i dati verranno caricati nel processo principale)

"num_epochs": int, # Numero di epoche di train
"eval_interval": int, # Intervallo di validazione
"prompts": {
    "use_boxes": bool, # Se True usa le boxe per il train
    "use_points": bool, # Se True usa i punti per il train
    "use_masks": bool, # Se True usa le annotazioni per il train
    "use_logits": bool, # Se True usa i logits dell'epoca precedente (se True viene ignorato use_masks)
},
"multimask_output": bool,

"opt": {
    "learning_rate": int,
    "weight_decay": int,
},

"sched": {
        "type": str = "ReduceLROnPlateau" or  "LambdaLR"
        "LambdaLR": {
            "decay_factor": int, # Fattore di dacadimento del lr funziona tramite la formula -> 1 / (decay_factor ** (mul_factor+1))
            "steps": list int, # Lista che indica ogni quante epoche deve decadere il lr (il primo step dev'essere maggiore di warmup_steps)
            "warmup_steps": int, # Aumenta il lr fino ad arrivare a stabilizzarlo in questo numero d'epoche
        },
        "ReduceLROnPlateau": {
            "decay_factor": float (0-1), # Fattore di dacadimento del lr funziona tramite la formula -> lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": int, # Pazienza per il decadimento del lr
            "threshold": float,
            "cooldown": int,
            "min_lr": int,
        },
    },

"losses": {
    "focal_ratio": float, # Peso di Focal loss sulla loss totale
    "dice_ratio": float, # Peso di Dice loss sulla loss totale
    "iou_ratio": float, # Peso di Space IoU loss sulla loss totale
    "focal_alpha": float, # Valore di alpha per la Focal loss
    "focal_gamma": int, # Valore di gamma per la Focal loss
},

"model_layer": {
    "freeze": {
        "image_encoder": bool, # Se True freez del livello
        "prompt_encoder": bool, # Se True freez del livello
        "mask_decoder": bool, # Se True freez del livello
    },
},

"dataset": {
    "auto_split": bool, # Se True verra usato il dataset presente in path ed effettuare uno split per la validation della dimensione di val_size 
    "seed": 42,
    "split_path": {
        "root_dir": str,
        "images_dir": str,
        "annotation_file": str,
        "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        "val_size": float (0-1), # Percentuale grandezza validation dataset
    },
    "no_split_path": {
        "train": {
            "root_dir": str,
            "images_dir": str,
            "annotation_file": str,
            "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        },
        "val": {
            "root_dir": str,
            "images_dir": str,
            "annotation_file": str,
            "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        },
    },
    "positive_points": int, # Numero punti positivi passati con __getitem__
    "negative_points": int, # Numero punti negativi passati con __getitem__
    "use_center": True, # Il primo punto positivo sarà sempre il punto più significativo per ogni maschera
    "snap_to_grid": True, # Allinea il centro  alla griglia di predizione utilizzata dal predittore automatico
}
```

Predizioni:
```python
"opacity": float,  # Trasparenza delle maschere predette durante la stampa dell'immagine
```

</details>

## Run model

Eseguire il file [`finestSAM/__main__.py`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/__main__.py)

Args (obbligatori):

```python
--mode (str)
```

### Train

```python
--mode "train"
```

### Predizioni automatiche:

```python
--mode "predict" --input "percorso/image.png"
```

* Args (opzionali - modificabili anche in config):

    ```python
    --opacity (float) default:0.9 
    ```

## Results

E' stato svolto un fine-tuning del modello per riuscire ad effettuare un'Instance Segmentation in maniera efficiente, generando poligoni delimitanti zone urbanistiche presenti in pdf rappresentanti strumenti urbanistici che regolano le trasformazioni del territorio 
(ad esempio, alcuni poligoni sono indicativi di perimetri entro cui sono validi alcuni vincoli edificativi)

Per questi test è stato ulizzato un solo prompt per l'allenamento, ossia `1 punto centrale per maschera allineato alla griglia del predittore automatico`, questo è risultato il prompt migliore per l'allenamento per il funzionamento del predittore automatico di SAM

### Test sam vit_b
<details>

<summary> Andamento del train </summary>

![Train sam vit_b](assets/Test-vit_b/Test6.png)
<p style="text-align: center; font-style: italic; width: 100%; font-size: small;">
 Andamento del train per il modello sam_vit_b
</p>

</details>

<details>

<summary> Immagini di comparazione </summary>

![Test 6 - Comparison 1](assets/Test-vit_b/Test6_comparison_1.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_b - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 2](assets/Test-vit_b/Test6_comparison_2.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_b - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 3](assets/Test-vit_b/Test6_comparison_3.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_b - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 4](assets/Test-vit_b/Test6_comparison_4.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_b - Maschere`_ , _`Finetuning - Maschere`_ 

</details>


### Test sam vit_h
<details>

<summary> Andamento del train </summary>

![Train sam vit_h](assets/Test-vit_h/Test8.png)
<p style="text-align: center; font-style: italic; width: 100%; font-size: small;">
 Andamento del train per il modello sam_vit_h
</p>

</details>

<details>

<summary> Immagini di comparazione </summary>

![Test 6 - Comparison 1](assets/Test-vit_h/Test8_comparison_1.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_h - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 2](assets/Test-vit_h/Test8_comparison_2.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_h - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 3](assets/Test-vit_h/Test8_comparison_3.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_h - Maschere`_ , _`Finetuning - Maschere`_ 

![Test 6 - Comparison 4](assets/Test-vit_h/Test8_comparison_4.png)
 _`Immagine originale`_ , _`Maschere di verità`_ , _`SAM vit_h - Maschere`_ , _`Finetuning - Maschere`_ 

</details>

## Possibili aggiunte 

- [ ] Allenamento funzionante anche per il predittore manuale:

    - [ ] Aggiunta di una funzione per predirre tramite il predittore manuale

    - [ ] Aggiunta di una funzione per creare le bbox per il train (suggerimento riga 175 [finestSAM/model/dataset.py](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/model/dataset.py))

- [ ] Metodo di validation basato su SAM automatic predictor

- [ ] supporto tpu

## Risorse

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

## License
The model is licensed under the [Apache 2.0 license](https://github.com/Marchisceddu/finestSAM/blob/main/LICENSE.txt).