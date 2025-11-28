This directory is for saving completed models.

There are 2 sub directories:
Sandbox (for working models.  Ignored by git)
final (for completed models).  They will be persisted in the repository

To move a model from sandbox to final, simply drag the whole directory over.
You will need to reference this new path later

Re-Running the model training tool with different variables makes a lot of files. We only need to save one
working, optimized, trained model, which is why this is set up this way. It lets you experiment with different model
parameters without making a mess, and lets you save only the model you need.
