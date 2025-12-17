# Automate Custom Events (Fork)

Questo repository è un **fork del progetto originale Automate Custom
Events di Pupil Labs**, esteso e modificato per supportare funzionalità
aggiuntive e miglioramenti operativi orientati a contesti di ricerca e
sperimentazione avanzata.

Lo strumento consente l'analisi automatica di registrazioni **Pupil
Invisible / Pupil Neon** tramite modelli di Intelligenza Artificiale, al
fine di rilevare **eventi personalizzati** basati sul contenuto visivo e
sulla posizione dello sguardo.

## Fork rationale

Il fork nasce da esigenze **scientifiche e di ricerca applicata** emerse
durante l'utilizzo del tool in contesti sperimentali reali, in cui:

-   Le analisi possono durare diverse ore o giorni.
-   È necessario poter **interrompere in modo controllato**
    l'elaborazione senza compromettere i dati già processati.
-   La riproducibilità e il controllo dell'esperimento hanno priorità
    rispetto alla semplicità dell'interfaccia.

In particolare, il progetto originale non prevedeva un meccanismo
esplicito di interruzione dell'elaborazione asincrona. Questo fork
introduce una gestione esplicita dello **stop dell'esecuzione**,
rendendo lo strumento più adatto a:

-   Studi longitudinali su grandi dataset egocentrici.
-   Prototipazione rapida di pipeline di annotazione automatica.
-   Validazione comparativa di modelli AI (cloud vs locali).
-   Contesti accademici o industriali in cui è richiesto maggiore
    controllo sul ciclo di esecuzione.

Le modifiche sono state progettate per rimanere **minimamente
invasive**, mantenendo compatibilità con l'upstream e facilitando futuri
riallineamenti.

## Differenze rispetto al progetto originale

Questo fork introduce in particolare:

-   Pulsante **STOP** nell'interfaccia grafica con gestione thread-safe
    dell'interruzione.
-   Migliore controllo dell'esecuzione asincrona.
-   Documentazione estesa e chiarimenti operativi.
-   Struttura compatibile con il repository upstream.

## Modelli supportati

-   **OpenAI (GPT-4o)** -- Alta precisione, richiede API Key.
-   **Google Gemini (2.5 Flash)** -- Veloce ed economico, richiede API
    Key.
-   **Local (CLIP)** -- Esecuzione locale, consigliata GPU.

## Requisiti

-   Python 3.9 o superiore
-   Token API Pupil Cloud
-   API Key del provider selezionato

## Installazione

``` bash
pip install .
```

Dipendenze aggiuntive per CLIP:

``` bash
pip install transformers torch pillow
```

## Avvio

``` bash
pl-automate-custom-events
```

## Utilizzo

Configurare la registrazione, il provider di modelli e la definizione
degli eventi tramite interfaccia grafica.\
Avviare con **Compute** e interrompere in qualsiasi momento con
**Stop**.

I risultati vengono salvati come `custom_events.csv` e, se configurato,
caricati su Pupil Cloud.

## Licenza e attribuzione

Questo progetto è un fork del software originale di **Pupil Labs**.\
Le modifiche rispettano i termini della licenza originale del progetto
upstream.
