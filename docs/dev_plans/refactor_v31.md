# Refactor spec
The current implementation has the right principles. Registries for displays, instruments, modalities, and optocontrols are all important. However, in practice it seems to still be leading to spaghetti code. For example, the optocontrols present a challenging design problem. I want them modularly separated with a registry and all, but the way a modality uses what an optocontrol provides varies between modalities. So it's nice to have abstraction, but I end up still coding things in my modality runner scripts like (if this optocontrol, then do this, if this optocontrol, then do this), which was the whole thing I wanted to avoid with modularity.

There's other problems too. I'm just brainstorming and word vomiting what I can think of:

The displays are a little weird. I have different data objects and different displays are compatible with different datas, so users can have multiple displays open at once if they so desire. The problem with this is that different modalities output different datatypes, and I'm making the fact that a display can display "one data object" a central god object.

Modalities themselves are a super mega god object, which I really don't like. When I change modalities, the displays don't self destruct (good), but it's possible that the modality I changed to is not compatible with the existing display, in which case the user needs to go add a display manually and set it up, which is annoying.

The partial solution to this is to have modality changing be a high stakes operation on the UX side of things, so it raises a dialog with what will change and the user has to confirm. Then the problem becomes making modalities actually properly distinct -  currently things like split stream and confocal are very similar, but with a few insidious differences that make it so that switching between them isn't just a button click. So I might be doing my regular imaging, and might want to switch back and forth between them. I agree that they shouldn't be distinct, that's something we need to think about. Indeed, even when imaging confocal versus flim - say I do my setup and image confocal, then go to flim, but realize something is wrong so I need to go  back to confocal. This workflow demands switching of modalities regularly, which might be an annoying user experience if switching modality is a high stakes process. Sure, once things are set up, the user won't be switching modalities - but the point still stands that the setup phase becomes irritating if modality changing is high stakes. 

I need a clean way of structuring the ability for feedback from UI to flow to the modality. One thing that I wanted to make was a system where I could draw a box on the display, then the modality goes and reacquires an image for just that box (for a particular modality where the idea of drawing-box-and-reacquiring is a thing). This presents a number of problems: how do I make it so that I can draw a box for any display? Do I need to have something that tells the user that the display is not drawing-box-compatible? And let's say that's not an issue, and by some miracle every display is box-drawing compatible. Then how do I have the feedback flow from display to acquisition in a way that doesn't violate modularity?

For instruments, I want instruments to be something that is an interface where instruments are all usable, with their connections testable. That's sort of there already, for the Swabian TimeTagger, but I'll eventually implement other instruments like the prior stage where I'll not only want control features aside from connection testing, but also features like the display constantly updating the location of the stage and more. And even features like autofocusing, for a modality where autofocusing is possible, maybe on one isntrument I have the capabilities for confocal imaging but not autofocusing, how does autofocusing get handled then?

These are all issues reflective of a deeper architectural problem. I tried to resolve it by organizing the UI towards the idea of routines, with modalities being a user facing thing, but the backend being routine oriented. But I'm not convinced that this solves the deeper underlying problems. 

## Proposed directory structure
```
- structs
  - parcel.py # just one parcel variety?
  - manifest.py
  - parameters.py 
- gui
  - File 1
  - File 2
  - Folder B
    - File 3
- Folder C
  - File 4
  ```

## Ideas
Have a manifest compatibility checker that warns the user with a dialog or halts acquisition if somethiings up (like no displayss configured to be enabled, or no modalities streaming even though the displays are configured to only want streaming, or whatever else). Ideally, this derives the compatibilities from implementation - perhaps a decorator on functions that do add_mask_optocontrol() that goes back to something that records things to the manifest, so that its autoassembled based on implementation instead of desired things to implement that I possibly forget.

The important consideration then becomes the data types. It is crucial that these are general enough that displays remain agnostic to acquisition types, and complete such that I don't need to make new data types as I implement new acquisition types.

As for modality behaviors, I'm thinking we have some modality routine that the user can configure. This is what we were getting at earlier. It's something we would need to actually create, but it centralizes all the choices that relate to what the modality actually ends up doing. For example, maybe the user does some stuff to set up confocal imaging, then makes a routine to state that we want to do optocontrol via scripts, with the feedback to draw a region on the UI to reacquire that region, and some other stuff. Then the necessary parameters given that routine pop up in one centralized location - this includes what I currently call parameters in the software, but also other things, like the channel to apply the script on, how to use the script, whatever else, and whether to even enable it for a particular imaging run. Those are things that would typically go in a different UI tab (the optocontrols tab), but now get centralized to the acquisition tab. 

Of course, the concern with such an implementation is that we need to be really careful, from a user experience side, how this routine thing works. 

## Current feature list
- Modalities
  - Confocal
  - Split confocal
  - FLIM (but it sucks)
  - Aspire: widefield, spectral
- Optocontrol
  - Masking
  - Aspire: scripts
- Instruments
  - Timetagger
  - Aspire: prior, zaber
- Displays 
  - 4 pyqtgraphs
- Data model (ditch)
- Coordination (also ditch)
- Persistence (ditch)
- GUI
  - PyQt6Ads (keep for sure)