# Detecting Owl Sounds

Here's how to detect powerful owl sounds in a lot of audio files. If you have ~2h of data, https://ninoxstrenua.site will do it in about five minutes. But if you have ~200h, or that site keeps crashing, you will need to follow these steps:

## Rent a LambdaLabs instance

Sign up to https://cloud.lambdalabs.com/ and put in your payment details. Attach a SSh key to your account. You can probably use another cloud GPU provider (or a gaming PC if you have one and are willing to mess around with setup), but this is the easiest path.

Go to https://cloud.lambdalabs.com/instances and click "Launch instance". You can select basically anything here but my experience of the <$1/hr instances is fairly buggy. I use 1xH100 or 1xA100. There's no need to attach a filesystem, but if you have a _ton_ of data and you want to keep it around, it'll probably save time.

It should show up in your instances with status "Booting". Note that **you are now paying for this**. Once you're done, you'll have to hit the checkbox next to the instance, then hit "terminate" in order to stop paying.

## SSH in and 
