## Your first Slurm job

As a reminder, the cluster is divided into two groups:

- **Compute nodes:** These actually perform the computation.
- **Login nodes:** These nodes you SSH into to submit jobs.

A job will typically be a single run of your experiment pipeline, or just one part of it. Basically, whenever you’d locally run `python train_models.py` or `python evaluate_results.py`, you’ll instead submit that command as a job. But of course you can bundle multiple commands into a single job as well.

Slurm will then put your submission into the *job queue*, and actually run your commands as soon as resources are available. For every job you submit, you need to specify its resource requirements (how many CPUs and GPUs you want, how much memory, and an upper bound on how long your job will take).

Let’s look at our first example. SSH into `rnn` and then run the following command:

```bash
srun --mem=100mb --time=0:01:00 cat /etc/hostname
```

This should show

```
ddpg.ist.berkeley.edu
```

(potentially with a different hostname). What you told Slurm using this `srun` command was:

> “I want to run the command `cat /etc/hostname`, I need 100 MB of (CPU) memory, at most 1 minute, and otherwise default resources."
> 

Slurm waited until those resources were free (which was probably immediately), then ran the command on one of the compute nodes (in this case `ddpg.ist.berkeley.edu`) and printed the output.

### Types of resources

For any Slurm job, you can specify how many resources you want. The available resources are:

- Memory (i.e. RAM)
- CPU cores (these are logical, i.e. hyperthreaded cores)
- GPUs (we have `A6000`, `A4000`, `A100-SXM4-80GB`, and `A100-PCI-80GB` GPUs)
- GPU shards, i.e. parts of a GPU

We’ll see how to specify these in the next section.

To elaborate on GPU shards: some jobs (e.g., deep RL training) might only need to use a GPU intermittently, so it’s a waste to allocate an entire GPU for a single job. Some of the GPUs on the cluster are split into *shards* which let multiple jobs use a single GPU. Each shard corresponds to 1GB of GPU memory, so if for example your job needs 6GB of GPU memory, request 6 shards. We currently do not enforce hard limits on the GPU memory usage of jobs requesting shards, but please request an appropriate number of shards to be courteous to other cluster users.

<aside>
⚠️ If you’re using `jax` or `tensorflow` with GPU shards, there’s an important extra step (if you’re using `pytorch` or not using shards, ignore this warning).

`jax` and `tensorflow` preallocate most or all of the GPU memory immediately for efficiency purposes (to avoid fragmentation). Since the sharding mechanism is only a way for Slurm to allocate GPUs and not actually visible to `jax` or `tensorflow`, they will by default allocate the entire GPU, as opposed to only the requested shards. You need to either disable preallocation or manually set the right amount of memory to preallocate. For example for JAX, you can put the following snippet at the beginning of you `sbatch` script (we’ll introduced `sbatch` later):

</aside>

```bash
TOTAL_MEMORY=$(nvidia-smi -i 0 --format=csv,noheader,nounits --query-gpu=memory.total)
# Assuming you requested 8 shards, otherwise adjust this.
# Preallocate 90% of requested memory. Allocating everything can lead to problems
# because e.g. the A4000 actually only has 16376 MiB available instead of 16384 MiB.
TARGET_MEMORY=$(echo "8 * 1024 * 0.9" | bc -l)
# scale=2 means two digits after the decimal point
FRACTION=$(echo "scale=2;$TARGET_MEMORY/$TOTAL_MEMORY" | bc)
echo "Allocating $FRACTION of GPU ($TARGET_MEMORY MiB out of $TOTAL_MEMORY MiB)"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION
```

### **Common `srun` arguments**

`srun` has a ton of arguments, but here are the ones you’ll use most frequently for simple tasks:

- Resources:
    - `--mem=1gb` The amount of RAM you need, if no unit is given MB is assumed*.*
    - `--time=hh:mm:ss` This should be an upper bound for how long your job will take. After this amount of time, the job will be killed immediately, without any warning! So usually, you’ll want to pick something a bit longer than your best guess. On the other hand, the more time you request, the longer it may take until your job will start. For any quota, only the actual time taken by your job counts, not the upper limit you specified.
    - `--cpus-per-task=4` or `-c 4` for short: How many CPUs you need. Default is 1. As the `per-task` part shows, things can get more complicated if you want multiple so-called “tasks” running for a single job, but we won’t cover that in this tutorial.
    - `--gpus=1` or `-G 1` for short: How many GPUs you need. Default is 0. You can also use e.g. `--gpus=A6000:1` to request a specific type of GPU. Unless you really need a specific GPU, we recommend not specifying a type since that gives the Slurm scheduler more flexibility and means your job might start sooner.
    - `--gres=shard:10` requests 10 *shards* of a GPU (see previous section). You can use `--gres=shard:<type>:<count>` like before.
- `--job-name=helloWorld` You can give names to your jobs, which can become useful once you have several of them running at once. But every job will also have a unique ID you can identify it with.
- `--output=filename` If set, the standard output of your command will be written to this file instead of printed to console.
- `--error=filename` Same as `--output`, but for standard error (i.e. error messages resulting from your command).
- `--qos=default` QoS, or Quality of Service, is essentially the priority of your job compared to others. If you want your job to run sooner, you can specify a higher priority, but you can only have a limited number of higher-priority jobs at any given time. Higher priority jobs can also access more resources. See the [QoS (Quality of Service) and prioritization](https://www.notion.so/QoS-Quality-of-Service-and-prioritization-dfc4d5660cc543b187ff2a4cb3d15cef?pvs=21) section below for details.

For all the arguments, the equals sign is optional (e.g. `--gpus 1` does the same as `--gpus=1`). Note that all arguments must come before the command you want to run! Otherwise, Slurm will interpret them as arguments to your command, instead of as arguments to `srun`.

### Running multiple commands

If you want to run multiple commands in a single job, you can use

```bash
srun --mem=1gb --time=1:00:00 bash -c "command 1; command2; command3"
```

This will run `command1`, then `command2`, and then `command3`. Note that later commands will run even if earlier ones failed. You can use `"command1 && command2 && command3"` if you want the entire job to abort once any command fails.

For realistic projects, you will usually have multiple commands for each job, and they’ll always be the same ones, so it makes sense to put them into a single bash script. In this example, you’d create `script.sh` (or some other name) on your NAS folder, make it executable with `chmod +x path/to/script.sh`, and put the commands in there:

```bash
#!/bin/bash
set -e
command1
command2
command3
```

(`set -e` has the same effect as using `&&` above, i.e. aborting if any command fails)

Then you can run

```bash
srun --mem=1gb --time=1:00:00 "/nas/ucb/<your_username>/path/to/script.sh"
```

You’ll need to pass the full path to your script, and the script has to be on the NAS, so that the compute node can access it.

## **Interactive Shell Sessions**

For any long-running training runs, you should submit commands like we’ve shown above (or typically using `sbatch`, which we’ll cover soon). But sometimes it’s useful to have an *interactive* session on a compute node instead, mainly for debugging.

To get an interactive shell on a node, use `srun` to invoke a shell:

```
srun --pty --mem 1gb --time=01:00:00 bash
```

The command we’re running is just `bash`. The `--pty` option is what lets you interact with the `bash` session like you normally would. Once you’re done, enter `exit` to return to the session on the login node.

<aside>
⚠️ **Please do not leave interactive shells running for long periods of time when you are not working. This blocks resources from being used by everyone else.**

</aside>

## `sbatch` and the job queue

When you run `srun`, you see the output of your command directly in your terminal, you can’t do anything else in that login session until your job is done, and if your login node session ends, the job is killed. For long-running commands like training runs, it’s usually more convenient to have them run in the background.

The `sbatch` command allows you to do just that. For our purposes, `sbatch`

- only accepts bash scripts, not inline commands.
- automatically redirects the output of your job to a file, similar to the `--output` option of `srun`.
- runs your job in the background, so you can do other stuff in the same login shell while waiting for the job to finish.
- Detaches the job from your login shell, so if the shell session ends, the job will still finish.

The actual technical differences between `srun` and `sbatch` are greater than this list suggests, but we’ll ignore that here.

Let’s look at a simple example. Create `helloWorld.sh`:

```bash
#!/bin/bash

srun echo "Hello World from" `hostname`

```

Then we can submit the job script. `sbatch` takes the same resource arguments you already know from `srun`:

```
sbatch --mem=1gb --time=1:00:00 helloWorld.sh
```

You should see something like

```bash
Submitted batch job 121
```

There are two things to note here:

- We have an `srun` command *within* the job script we’re submitting. Explaining this goes beyond the scope of this tutorial, but the practical version is: in your `sbatch` job scripts, it’s generally good practice to run things via `srun` (and it has some benefits, like letting you see later how long each command in your script took). But for simple scripts like the ones we’ll be using, everything will still work fine without `srun` commands.
- We didn’t provide the full path to the script this time (and it doesn’t actually have to be on the NAS, and doesn’t have to be executable). The reason is that `sbatch` handles scripts differently than `srun`. For `srun`, the script was just like any other command, so it had to be able to find it and it had to be executable. In contrast, `sbatch` *only* accepts job scripts, and it will automatically copy them to the compute node and run them there. (This also means after you’ve submitted a job, changing the job script doesn’t have any effects.)

### Inspecting job status

You can check on the status of your job using `squeue`:

```
$ squeue
 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
   121      chai helloWor username  R       0:01      1 chai
```

(Though when you run that command now, the job might not actually show up because it has probably finished already.) The `R` means your job is currently running, and `TIME` shows you how long it’s been running for. `squeue` shows all your currently running and scheduled jobs. Sometimes, jobs will start running immediately after you run `sbatch`, but if the required resources aren’t available, you might have to wait a while.

<aside>
ℹ️ Add the two lines below to your `~/.profile` file and then run `source ~/.profile` to show more useful columns in the output of `squeue`:

```bash
export SQUEUE_FORMAT2="JobID:.8 ,Name:.20 ,QOS:.9 ,UserName:.8 ,StateCompact:.2 ,TimeUsed:.12 ,NodeList:.25 ,Tres:.50 ,Reason"
unset SQUEUE_FORMAT
```

</aside>

The output of your job will be written to a file in the directory you run `sbatch` from, in a file called `slurm-<job id>.out`. This file is already accessible while the job is running, so if you want to monitor the output of a currently running job, you can do

```bash
tail -n 20 -F slurm-<job id>.out
```

This will show the last 20 lines of the job log and continually update them. You can even run this command before the job log exists and it will keep trying, which can be nicer than spamming `squeue` to see whether your job has started. In practice though, this should mostly be for debugging purposes. If you find yourself monitoring every job’s output or checking all the time whether your job has started, then maybe you could have just used `srun` instead, or maybe you should set up a more scalable logging solution like Weights & Biases or Tensorboard.

Like for `srun`, you can change the location of the job log file with `--output=filename`. There are also variables you can use, e.g. `--output=folder/my-job-%j.txt` will save to `folder/my-job-<job number>.txt`. The other variable that can be useful is `%x` for the job name (which you can set with `--job-name`).

### Cancelling jobs

For jobs started with `srun`, you can cancel them simply using `Ctrl+C` but that won’t work for `sbatch` since the job isn’t running in your shell. Instead, there is a dedicated `scancel` command. The usage is simply

```bash
scancel <job id>
```

You’ll see the job id right after submitting a job via `sbatch` but can also get the ids of all your running job from `squeue`.

### Arguments in job scripts

If you have a job script, the resource requirements for that job are probably fixed, so it would be annoying to put them into the `sbatch` command every time. Because of that, `sbatch` allows you to put arguments in special comments within your job script instead. For example:

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --gpus=1
#SBATCH --time=01:00:00

srun command1
srun command2
```

## QoS (Quality of Service) and prioritization

QoS is how we manage prioritization, i.e. which job gets started first if there aren’t enough resources for all jobs in the queue. You can specify the QoS for a job with the `--qos` flag. The current QoS levels from highest to lowest priority are:

- `high`: use this for interactive jobs, debugging, results you need urgently, etc. Users are limited to one node’s worth of resources over all their running high priority jobs: 256 CPUs, 8 GPUs, and 1024GB of memory.
- `default`: use this for batch jobs that aren’t as urgent. Users are limited to 32 GPUs over all their running default priority jobs, but there is no limit on CPUs/memory/shards.
- `scavenger`: use this when the cluster is empty to run jobs without being *billed* for them (see [Billing](https://www.notion.so/Part-2-Using-Slurm-538d4cc4f0fa47f0869bf92d0d22ad9e?pvs=21) section below). **Important:** running jobs on this queue requires also specifying `--partition scavenger`. The same resource limits apply as the `default` QoS. See below for information about preemption for `scavenger` jobs.

Note that you can always *submit* as many jobs as you want, they just won’t all run at the same time if they exceed the resource limits for the QoS.

### Preemption

Jobs with the `scavenger` QOS can be *preempted* by jobs with the `default` or `high` QOSs. `scavenger` jobs are always guaranteed to run for one hour once started. However, after an hour, if there is another job with a higher QOS that requires a `scavenger` job’s resources, it will be preempted.

When a job is preempted, it is either *requeued* or *canceled.* The job will be requeued if the `--requeue` option is passed to sbatch; this means that it is stopped and put back in the job queue, losing any intermediate state. If `--requeue` was not passed to sbatch, then the job is canceled. 

### Billing

In order to make sure that the cluster resources are fairly allocated among all users, we use fair-share scheduling. When a job is running, it is *billed* based on the amount of resources it requests times the amount of time it is running. If you’ve added [these two lines](https://www.notion.so/Part-2-Using-Slurm-538d4cc4f0fa47f0869bf92d0d22ad9e?pvs=21) to your .profile, then you’ll see the amount the job is being billed under the `TRES_ALLOC` column:

```bash
   JOBID                 NAME       QOS     USER ST         TIME                  NODELIST                                         TRES_ALLOC REASON              
  378821                 bash   default  cassidy  R         0:05      vae.ist.berkeley.edu        cpu=20,mem=64G,node=1,billing=68,gres/gpu=1 None      
```

For example, this job is being billed at a rate of 68. The billing rate is a weighted sum of the resources being used:

```bash
billing = 1*cpus + 0.25*mem_gb + 32*gpus + 1*shards
```

Users with higher resource usage over the past several days have lower priority compared to other users when submitting jobs to the same QoS, so don’t request more resources than your job needs to avoid your priority being lowered. To see all users’ resource usages, run the command `/nas/ucb/cassidy/bin/allshare`:

```bash
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare 
...
  chai                  cassidy          1    0.008929   376056877      0.235320   0.008850 
...
```

For each user, the `EffectvUsage` column shows the percentage of the total cluster resources that user has recently used: in this case, Cassidy has used around 23.5% of the cluster resources over the past several days. The `FairShare` column shows the priority weight assigned to each user based on their resource usage. A higher `EffectvUsage` leads to a lower `FairShare`.

As described in the QoS section, the `scavenger` QoS can be used to run jobs without being billed. However, it has lower priority than the `default` queue and can be preempted, so if the cluster becomes full you may need to move scavenger jobs to default priority!

## More documentation

The full SLURM documentation is available [here](https://slurm.schedmd.com/). Feel free to ask questions in the #compute channel on Slack!

Next up: [Part 3: Example walkthrough for a Python project with Slurm](https://www.notion.so/Part-3-Example-walkthrough-for-a-Python-project-with-Slurm-8a37245f2f2e41bba0bad06b86b791d4?pvs=21)