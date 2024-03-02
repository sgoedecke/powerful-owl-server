# Detecting Owl Sounds

Here's how to detect powerful owl sounds in a lot of audio files. If you have ~2h of data, https://ninoxstrenua.site will do it in about five minutes. But if you have ~200h, or that site keeps crashing, you will need to follow these steps:

## Rent a LambdaLabs instance

Sign up to https://cloud.lambdalabs.com/ and put in your payment details. Attach a SSH key to your account. You can probably use another cloud GPU provider (or a gaming PC if you have one and are willing to mess around with setup), but this is the easiest path.

Go to https://cloud.lambdalabs.com/instances and click "Launch instance". You can select basically anything here but my experience of the <$1/hr instances is fairly buggy. I use 1xH100 or 1xA100. There's no need to attach a filesystem, but if you have a _ton_ of data and you want to keep it around, it'll probably save time.

It should show up in your instances with status "Booting". Note that **you are now paying for this**. Once you're done, you'll have to hit the checkbox next to the instance, then hit "terminate" in order to stop paying.

## SSH in and set up

Click to copy the text in "SSH Login" (it should be something like `ssh ubuntu@104.171.202.77`). Open a terminal in your computer and paste that in. If you've set up your SSH key correctly, it should prompt you with "Are you sure you want to continue connecting". Type "yes", hit enter, and you'll be onto the server.

```
git clone https://github.com/sgoedecke/powerful-owl-server
cd powerful-owl-server/manual/
```

Paste these commands in to install dependencies:

```
pip install soundfile librosa evaluate transformers pydub
pip install accelerate -U
```

Now you need to copy your files over to the server. Open a **new** terminal (not the one you've been using), navigate to where your files are, and run a command like this:

```
scp -r 103_BoodjamullaLawnHillNationalParkDryB ubuntu@104.171.202.77:owls
```

You should use the directory where your files are instead of `103_BoodjamullaLawnHillNationalParkDryB`, and whatever your LambdaLabs SSH command was instead of `ubuntu@104.171.202.77`. Depending on your internet speed and the size of the files, this copy may take some time (e.g. 5-10 minutes).

(If you did attach a filesystem earlier, here's where you'd specify that instead of `:owls`, or where you'd copy the files to after they're done uploading.)

## Look for owls

Now you get to look for owls! Go back to your original terminal (the one SSHd into your server, that should say something like `ubuntu@104-171-202-77:~/powerful-owl-server/manual` in the prompt), and run this:

`python ./infer.py ~/owls`

There's going to be a lot of text on the screen, some of which will say "warning". Don't worry about it.

At the end, it should print out a results block like this:

```
---------------------------------------------
Potential owls found:
File: /home/ubuntu/owls/20210603T080000+1000_Boodjamulla-Lawn-Hill-National-Park-Dry-B_911228.flac
  Start: 00:02:00, End: 00:02:05
  Start: 01:52:40, End: 01:52:45
```

That shows the owl sounds that the model found. Bear in mind that many of these are likely to be false positives, given the current prototype state of the model - you should listen to them to confirm. But hopefully it lets you rule out tens of hours of data without having to listen to it.