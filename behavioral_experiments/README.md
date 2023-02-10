# Running experiments

To run our experiments, several things are needed:

* A web server with a public IP address to host the experiment for mturk participants.

* A GPU that is either already on the web server or can be securely tunneled into from the web server.

* A mongo database set up for writing data (optional)

Check-list before running expeirment:

1) navigate to `/behavioral_experiments/` and run `npm install` on web server to install dependencies, call `unzip ./src/local_imgs.zip`, then `node run build` to launch experiment

2) call `store.js` to launch database process on mongo port

3) navigate to `/models/` and run `python listen_on_gpu.py` on machine with GPU

4) if GPU and web server are on different machines, call `ssh -fNR 5003:localhost:5003 <web-server-domain>` on the GPU machine to establish a secure tunnel.

# Experiment options

In `config.json`, you can set "speakerType" and "listenerType" options to either 'human' or 'AI' to specify whether to run a multi-player 'human-human' experiment connecting a human speaker and human listener, a 'model-as-speaker' experiment, or a 'model-as-listener' experiment.

The 'contextType' option specifies whether to use 'easy' or 'hard' contexts.