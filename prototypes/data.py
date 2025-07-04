# This file will contain the sythetic video and mask generation logic

# I'll generate a video with random noise confined within a sample brightness range (I can use a simple gaussian distribution). Then I'll add glare in.
# I can track where I add the glare so I can determine what a perfect mask would look like. That way, I can provide really good feedback about what the mask should be.

def generate_sample_video(id):
    # according to the specifications I laid out, I'll create a piece of sample data.
    # id will be a sequential identifier for the video generated. I'll use this id both as a seed for any randomness, and for the name of the video
        # I can use either pytorch or numpy to create the seed


    # I'll create both a video and an "optimal" mask, both of which will have the same id
    # I may need to create a mask object or datatype.

    # True if no issues while generating, else false.
    return True