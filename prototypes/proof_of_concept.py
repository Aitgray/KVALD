# Full proof of concept script
# Imports

# Initial function
def video_processing(path):
    # For now this will just have a path to some kind of video file

    '''
    Initial processing steps:
        First and foremost, we'll need to check the file type to determine if it's a format we want it in. I'm not sure what format would be optimal. If it's not in our optimal format, we convert it if we can.

        The videos shouldn't have audio but if they do, we can immediately dump that data as it doesn't provide any value.
        
        Start by converting it to grayscale or something similar, I don't know how to extract the brightness from each frame but I can figure that out later. Along with the conversion to grayscale.

        Whatever resolution it is, we'll probably want to scale it down. We don't want to drop it too low because we don't want to lose detail, we'll want to ensure that we keep the hotspots when we scale down.
        I suspect that we can drop things to at least 720p, perhaps even 480p

        I'm not sure if there's any other processing steps I'll need to take.
    '''

    # This will return the processed version of the video, which is fine for this prototype. Eventually it'll take a video input and break it into frames in order to generate the mask.
    pass

# Mask generation function
def generate_mask(video):
    # This function will take the processed video and generate a mask
    # Start by breaking the video into frames and processing each individually.
    
    # We'll do this by feeding each frame into the neural network, and it'll generate a mask for each individual frame.
    # We'll then send each frame of the mask onto the next few functions. First we'll send it to the kalman filter and then we'll smooth it after it's done processing
    return "dummy_mask"

# Kalman filter application function
def apply_kalman_filter(video, mask): # I need to do more research to understand this. I think it may be unnecessary.
    # This function will apply the Kalman filter to the video using the generated mask

    return "kalman_output"

# Smoothing function
def smooth_output(kalman_output):
    # This function will smooth the mask that we have
    # The intention of this smoothing is to take a wider view, saving a few of the previous frames that we've interpreted and ensuring that our mask isn't flickering or jagged.
    # If we want to use this for driving, we must ensure that it's not a hinderance, that it improves visual clarity as opposed to harming it.

    return "smoothed_output"

# Feedback verification function
def verify_feedback(smoothed_output):
    # This function will verify the feedback from the smoothed output
    # This is kinda just a sanity check, is the processed mask that we've output good according to our heuristics?

    return "feedback_verified"

# Finalized output function
def finalize_output(verified_feedback):
    # This function will finalize the output based on the verified feedback

    return "finalized_output"

# Main function to run the proof of concept
def run_proof_of_concept(video_path):
    # Step 1: Process the video
    processed_video = video_processing(video_path)
    
    # Step 2: Generate a mask
    mask = generate_mask(processed_video)
    
    # Step 3: Verify feedback: measure our output based on our heuristics, ensure the validity of the mask and the model.
    verified_feedback = verify_feedback(mask)
    
    # Step 4: Finalize output: if we need to make adjustments to the mask based off the verified feedback, we'll perform them here.
    # Perhaps we check some kind of accuracy metric, if it's below a certain threshold we send it into the function
    if accuracy < 0.95: # accuracy is currently a placeholder
        finalized_output = finalize_output(verified_feedback)
    else:
        finalized_output = mask
    
    # Output the mask applied to the video so I can visually confirm it works
    # Print any useful extraneous data/statistics
    # Print the amount of time it took in total, for each frame on average, as well as outlier frames.
    # I can use a box plot to see the fastest frame time, as well as the slowest.
    # If I use multiple threads, I can check to see the active time of each, this way I can determine thread allocation (do certain processes require more/less threads than others).

    return 1