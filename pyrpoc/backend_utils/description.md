I don't like that idea. Fundamentally, opto-control is tied to acquisition timing, so no matter what, if we have a mask or something entered, acquisition will need to do something about that, which typically is more involved than just calling an extra function.

I think we don't need to worry about this. Let's just, for now, pretend that RPOC is only the DAQ mask type, which we can implement with ifs. We can change the backend structure later if this becomes problematic. Now let's think through how we need to actually code out the regestries for each of the four items. For each of the 4 (optocontrol, display, modalities, instruments), i want to code 3 things fully out now (at least to a point where i can get the GUI up and running): the registry, base_item, and simulated thing (except for opto control, where we will can probably just make the real thing). 

before we get to coding, let's plan out exactly what we want for each of those, and then figure out how we want to implement it. among other things, we will need to change the laser_modulation stuff to just rpoc. 
- modality
   - the modality is partially coded already. we need to say what parameters and set them up, say what instrumeents are required/optionally usable, say the output datatype for a single acquisition for any given modality type. so base_modality needs to enforce that each modality makes this. 

- instruments
   - each instrument will need a config and a control thing. when the GUI somehow prompts for addition of instrument X, there needs to be a way to connect (given config parameters), and also be a way to control. im unsure of exactly what the base instrument needs and what the GUI could read here, but im thinking that in the same way we have the GUI set up from acquisition parameters, we could have the instruments set up.
  

- displays
   - each display should actually be a qtwidget that we can host in the GUI - note the difference here from other things (base display will be a qtwidget). displays should be agnostic to the structure of modalities in the same way that instruments are agnostic to modalities. 
  
for now, don't think on optocontrol. lets get everything else set up first, and then worry about that part. further flesh out exactly what we need to put for each of those 3 to get a functional thing. dont worry about how the GUI looks, but think about what the GUI needs from each to be able to work agnostically knowing only the structure of the base (which I think is a bare minimum?)
