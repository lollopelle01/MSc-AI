\begin{center}
{\Huge Cognition and Neuroscience}
\end{center}

\tableofcontents

\newpage

# 1 Cognitive Neuroscience: Bridging the Gap Between Brain and Mind

## 1.1 Analyze the relationship between neuroscience and artificial intelligence. To what extent is the brain a good model for machine intelligence, and what are the potential benefits and limitations of using neuroscience to inspire AI development?

AI and neuroscience have a bidirectional relationship in which these two fields (both not yet fully understood) can inform each other. Neuroscience can provide inspiration for new types of algorithms and architectures not yet discovered, as well as validation for existing AI techniques if similar mechanisms are found to be implemented in the brain. Conversely, AI can provide insights into brain function, as seen in the example of understanding the prefrontal cortex as a meta-reinforcement learning system.

The brain is a partial but valuable model for AI. In its favor, it serves as existing proof that general intelligence is possible, and the "all-or-none" firing of neurons is analogous to binary computation. However, there are significant differences: brains and computers operate on fundamentally different principles. Unlike the distinct hardware and software of a computer, the mind and brain are not separate entities and influence each other reciprocally. Moreover, brains process information through biochemical changes and recurrent, cyclical circuits, not just linear causal chains.

Additionally, while machine learning often relies on statistical learning from vast datasets due to large memory capacity, the brain has limited memory yet excels at generalizing knowledge from limited data and transferring it to novel domains.

## 1.2 Discuss the historical development of the concept of localization of function in the brain. How has this concept evolved over time, and what evidence supports or challenges it?

In many different times in history the localization of function in the brain was discussed:

- In ancient times (500 BC) there were debated between cardiocentrism, which considered the heart as the mind's seat, and the encephalocentrism, where the brain was the mind's seat. The last one followed from the observation that brain damages affected mental abilities.
- In the 1800 was proposed phrenology, the pseudoscience (proven only with correlation approach) where through a careful analysis of the skull was possible to infer the personality. This followed from the hypothesized that if a person used one of the faculties with greater frequency than the others, the part of the brain representing that function would grow, causing a bump in the skull. This was the beginning if localizationism, where the brain is considered the organ of the mind and that innate faculties were localized in specific regions of the cerebral cortex.
- In the early 1800 was used a causal approach to prove phrenology but it was only proven that certain brain areas were responsible for certain functions but not for the advanced ones. So it was the beginning of the aggregate field theory, that multiple brain areas participate together for mind functions. It was unclear if only one was true or both.

  - In favor of localizationism, it was discovered a topographic organization of the cortex, studying epilepsy progression.
  - Still in favor of localizationism, it was reported that specific left-hemisphere lesions caused pecific language deficits (expressive and receptive aphasia).
- In the end of 1800 was proposed Cytoarchitectonics, the study of cellular architecture or how cells differ between regions, since it was strongly believed that different brain regions address different functions then they also must have different structures.

Nowadays we know that both localization and distributed processing are true, recognizing that while specific functions can be localized, complex cognition involves networks across the brain.

## 1.3 Structure and function are intimately related in the nervous system. Choose an example from cognitive neuroscience to illustrate that function or cognition directly emerges from the nervous system or brain structure.

Motor functions provide a clear illustration of the intimate relationship between brain structure and function.

In the motor system, movements of different body parts are mapped onto specific regions of the primary motor cortex. This organized mapping shows that motor function directly emerges from neural structure: damage to a specific cortical area produces deficits in the corresponding movements.

However, when motor areas are damaged, the function does not necessarily disappear permanently. After an initial impairment, recovery can occur through functional reorganization. Other cortical regions, such as neighboring motor areas or the contralateral hemisphere, can be recruited to support the lost function. This happens through changes in synaptic strength and connectivity, not through the creation of entirely new structures.

This shows that while motor function is initially constrained by anatomical structure, functional demands can reshape how existing structures are used. Important functions like movement drive the redistribution of neural resources, allowing partial or substantial recovery.

Therefore, motor control demonstrates a deeply intimate relationship: function emerges from structure, but sustained functional necessity can reorganize that structure over time.

# 2 The Nervous System: anatomy and physiology

## 2.1 Discuss the roles of glial cells in supporting neuronal function, including structural support, immune support, nourishment, and signaling. Provide specific examples of how different types of glial cells contribute to each of these functions.

Glial cells is a family of cells that play an essential role in supporting nerve cells in CNS, where they outnumber neurons, and in the PNS. The examples of cells are:

1. **Oligodendrocytes** (CNS) and **Schwann cells** (PNS) produce myelin, a white insulating layer that wraps around the axon, the neuronal structure responsible for transmitting signals to other neurons. This insulation prevents signal degradation and allows for saltatory conduction, where action potentials “jump” between gaps in the myelin sheath called Nodes of Ranvier, greatly speeding signal propagation. The more layers are applied and the more rapid and reliable the communication over long distances is, up to 2 meters in the PNS.
2. **Astrocytes** are star-shaped glial cells that surround neurons and closely associate with brain blood vessels. They help nourish neurons by regulating extracellular ion and neurotransmitter concentrations and facilitating nutrient exchange. Additionally, they play a key role in maintaining the blood-brain barrier, providing crucial structural and metabolic support that protects the CNS from harmful substances in the bloodstream.
3. **Microglial cells** provide immune support as they are immune system cells, so they identify when something has gone wrong and initiate a response that removes the toxic agent and/or clears away the dead cells.

## 2.2 Describe the structure of a neuron and explain how each component contributes to the neuron's ability to receive, process, and transmit information.

A neuron is a specialized eukaryotic cell designed for rapid communication. Like all eukaryotic cells, it contains:

- Cell membrane: separates intracellular and extracellular spaces and regulates ion flow.
- Cytoplasm: the intracellular fluid, rich in ions (especially K+, Na+, Cl-, Ca) and proteins. At rest, the interior is negatively charged (~ –70 mV) relative to the outside as there is more K.
- Extracellular fluid: positively charged relative to the cytoplasm due to higher concentrations of Na+ and Cl-.
- Cell body (soma): the metabolic center containing the nucleus and organelles, essential for protein synthesis and cellular maintenance.

In addition, neurons possess three specialized structural components that enable them to receive, integrate, and transmit information:

1. **Dendrites**:

   These are branched, tree-like extensions that form the input zone of the neuron. They receive chemical or electrical signals from other neurons via synapses located on their surfaces. Their extensive branching (arborization) maximizes the surface area for synaptic contacts, allowing the neuron to collect information from thousands of other neurons.
2. **Axon hillock**

   The final part of soma where the signals are integrated and trigger the postsynaptic potentials (PSPs) spike, if they reach the threshold (~ –55 mV).
3. **Axon**:

   A single, elongated projection that serves as the conductive zone. The axon then propagates this all-or-none electrical signal rapidly and without decrement toward the axon terminals.
4. **Synapses**:

   Located at the ends of axon terminals, synapses form the output zone. They are specialized junctions where the neuron communicates with other cells (neurons, muscles, or glands). At chemical synapses, action potentials trigger the release of neurotransmitters, which cross the synaptic cleft and bind to receptors on the postsynaptic cell, continuing the signal.

## 2.3 Explain the process of signal transduction within a neuron, from the reception of input signals at the dendrites to the transmission of output signals at the synapses.

The process is:

1. **Input**

   The dendrites receive in input from neurons or other cells a signal. The signal can be chemical (e.g., neurotransmitters from a neighboring neuron), electrical (via direct gap junctions), or physical (e.g., mechanical pressure or light in sensory receptors) and is handled differently:

   - **Phisical**

     	Stimuli like light, sound, pressure, or temperature are detected by specialized sensory receptors. These receptors contain ion channels that open or close in response to the physical stimulus, directly generating a localized change in membrane potential called a receptor potential.

   - **Electrical**

     	At a minority of synapses called gap junctions, neurons are directly connected by protein channels. An electrical signal (a change in membrane potential) in the presynaptic cell passes instantly and bidirectionally to the postsynaptic neuron through these channels, allowing for very fast, synchronized communication.

   - **Chemical**

     	Neurotransmitters released from a neighboring neuron bind to specific receptor proteins on the dendritic membrane. This binding causes ion channels to open, leading to a local, graded change in membrane potential known as a Postsynaptic Potential (PSP). It can be excitatory, increasing the charge, or inhibitory, decreasing it.

   Regardless of origin, all these inputs result in local, graded potentials that spread passively and decrementally toward the cell body. Those potentials are graded in amplitude and duration. 

2. **Integration**

   The signals then passes through the cell body and the axon hillock and between those two structures they are integrated. Here occurs the membrane potential cycle to make the neuron fire or not:

   - 2.1 **Resting**

     	The difference between cytoplasm and extracellular fluid, due to different concentrations of Pothassium (K+) and Sodium (Na+), is about -70mV and this is the resting potential at the beginning of the cycle.
   
   - 2.2 **Stimulus integration**

		The various inputs are passed passively through cytoplasm and so suffer from decremental conduction (max 1mm, way weaker than axonic). Anyway they are then summed up in axon hillock both spatially (from different synapses) and temporally (at different times). If the total voltage is $\le$-55mV then it is a failed initiation and it is ignored, otheriwse we proceed with depolrization.

   - 2.3 **Depolarization**

     	After reaching the threshold, the voltage-gated Na+ channels open allowing them to enter cytoplams from the extracellular fluid. This causes a quick increase in the membrane potential that comes up to +30mV. At that value we reach the Na+ equilibrium and so the channels become inactive and then we have repolarization.

   - 2.4 **Repolarization**

     	After the Na+ channels become inactive, the voltage gated K+ channels open allowing them to enter the cytoplasm decreasing the membrane potential. At this point we have an absolute refractory period, where no input could trigger the depolarization as the Na+ channels are inactive and the voltage decreases up to K+ equilibrium potential that is even below the resting potential, around -80mV. At this point we then have hyperpolarization.

   - 2.5 **Hyperpolarization**

     	At -80mV the Na+ channels reactivate so we face the relative refractory period in which the Na+ channels are opened and so depolarization is triggerable but the potential is still lower than the resting one so we'll need a stronger stimulus to trigger it. At this point K+ channels close and the ions pump helps the memebrane to return to resting potential by pumping in more sodium than the potassium pumped out.

   At this point we generated an action potential (AP), a signal of fixed amplitude (-55mV) and all the exceeding amplitude of input PSP will increase the frequency, together with the duration of the inputs.

3. **Conduction**

   The action potential is conducted through the axon where it is ensured to travel far and fast thanks to some features.

   - AP travels far since:
		- Only meaningful information are propagated
		- The conduction isn't decremental but rather active by self-regenerating along the portions of axon. As the current passes through a segment, AP depolarizes the next portion and hyperpolarizes the previous one.
		- As follows from the previous point, the hyperpolarization prevents the current from flowing back, ensuring that the current flows from dendrites to synapses.
		- Since the information isn't stored in the amplitued but rather in the frequencies, it is more robust to damage.
   - AP travels fast thanks to the myelin layers that isolate portions of the axons. In this way the current does not have to be regenerated at each point but only in the gaps left by myelin, so called nodes of Ranvier, prioducing the so called saltatory conduction.

4. **Output**

   The AP so arrives to the synapses where it is handled differently depending on the type of synapse:

   - **Electrical**

		The AP is directly transmitted through the gap junction between the synapses and the postsynaptic neuron thanks to cytoplams continuity enabled by pores. It is a less plastic synapse as it ca only reproduce the signal and there are not intermediate steps.

   - **Chemical**

     	The AP modulates the release of neurotransmitter following this process:

		- The AP reaches the presynaptic terminal and opens the voltage-gated calcium (Ca) channels allowing it to enter from the presynaptic cleft.
		- The higher concentration of Ca makes the vescicles of neurotransmitter fuse with presynaptic membrane, releasig neurotransmitter in the synaptic cleft.
		- The neurotransmitter makes the Na+ channels open in the postsynaptic cell, changing the potential.

     	This is more plastic as the impact of the presynaptic neuron depends on how the neurotransmitter is released by the postsynaptic neuron. Allowing the same inputs to possily have totally opposite outputs.

## 2.4 Compare and contrast the characteristics and functions of postsynaptic potentials (PSPs) and action potentials (APs). Explain how PSPs contribute to the generation of action potentials and how action potentials enable long-distance communication within the nervous system.

Postsynaptic potentials (PSPs) and action potentials (APs) are two fundamental types of electrical signals in neurons, each playing distinct yet complementary roles in neural communication. PSPs are graded potentials that arise at the dendrites and soma in response to synaptic input. Their amplitude varies with the strength of the stimulus, and they propagate passively and decrementally, meaning they weaken with distance as they travel through the cytoplasm. PSPs are primarily responsible for receiving and integrating information from other neurons, and they come in two forms: excitatory PSPs (EPSPs), which depolarize the membrane and move it closer to the firing threshold, and inhibitory PSPs (IPSPs), which hyperpolarize the membrane and inhibit firing.

In contrast, action potentials are all-or-none, non-decremental signals generated at the axon hillock once the summed PSPs reach a critical threshold. APs have a fixed amplitude and brief duration, and they propagate actively along the axon without losing strength. Their main function is long-distance signal transmission, ensuring that information travels reliably from the neuron’s input zone to its output synapses.

For the explanation see [previous point](#2-3).

## 2.5 Explain the significance of the "all-or-none" nature of action potentials. How do neurons communicate information about the strength or intensity of a stimulus given this limitation?

The all-or-none nature of APs guarantees a robust, non-decremental signal suited for long-distance communication, but it limits the neuron’s ability to encode stimulus strength in signal amplitude. To overcome this, the nervous system uses frequency and duration of firing—along with population recruitment—to translate graded sensory or synaptic input into a meaningful neural code. In this way, a digital-like pulse code (APs) can faithfully represent an analog world of varying stimulus intensities.

For the explanation see [answer 2.3](#2-3).

## 2.6 Discuss the significance of myelin in neuronal signaling. Explain how myelin is formed, how it contributes to the speed and efficiency of action potential propagation, and what happens when myelin is damaged in diseases like Multiple Sclerosis.

Myelin allows saltatory conduction. For the explanation see [answer 2.3](#2-3).

In Multiple Sclerosis (MS), an autoimmune disorder, the immune system mistakenly attacks oligodendrocytes and the myelin they produce in the CNS. This results in demyelination—damage and loss of the myelin sheath—and subsequent scarring (sclerosis). As a consequence:

- Saltatory conduction is impaired: action potentials can no longer jump efficiently between nodes. They must propagate continuously along demyelinated segments, which is much slower and requires more frequent regeneration of the signal.
- Signal slowing and failure: the slowed conduction delays neural communication. In severe cases, action potentials may fail to propagate altogether (conduction block), especially if the demyelination is extensive or if the axon’s ability to regenerate the signal is compromised.
- Increased vulnerability to signal disruption: without insulation, the axon is more susceptible to ionic imbalance, energy depletion, and interference, making neural transmission less reliable.

Clinically, this leads to symptoms such as muscle weakness, sensory disturbances, visual problems, and fatigue, depending on which neural pathways are affected. Thus, myelin is essential not only for speed but also for the fidelity and reliability of high-frequency signaling in the nervous system, and its loss in conditions like MS directly undermines these functions.

## 2.7 Compare and contrast electrical and chemical synapses, focusing on their structural differences, mechanisms of signal transmission, and functional implications for neural communication.

See a full explanation at [answer 2.3](#2-3).

## 2.8 Describe the components of a neural circuit, including sensory neurons, interneurons, and motor neurons. Provide an example of a simple neural circuit, such as the knee-jerk reflex, and explain how it functions.

> A neural circuit is a group of interconnected neurons that process specific kinds of information. No complex behaviour is initiated by a single neuron.

Each behaviour, for example considering phisical responses, is described by a circuit:

1. Sensory inputs: they carry the information from the periphreral sensor to the nervous system.
2. Interneurons: they mediate impulses between sensory and motor neurons.
3. Motor neurons: carry the response from the CNS to the to muscles and glands.

Each component is then mediate by a group of them or several groups of them, this last one option has evolutionary advantage as it implements a parallel processing that allows to simultaneously encode different information relative to the same stimulus.

Moreover based on how the components are wired we can identify two properties:

1. Divergence (1-M): observed more frequently at the input stages, a single neuron can exert multiple neurons.
2. Convergence (M-1): observed more frequently at the output stages, multiple interneurons concurr in activating a single motor neuron. This ensure it is activated only if a sufficient number ofsensory neurons become activated together.

Moreover the structure of components in a circuit can be slightly more complex than simply linear, for example:

1. Feed-forward inibition: excitatory neurons synapse ontoinhibitory interneurons, inhibiting other downstream neurons. In this way we enhance the mutual exclusion of opposite pathways.
2. Feed-back inibition: excitatory neurons synapse onto inhibitory interneurons, which project back to the same neurons and inhibit them. In this way a stimulated pathway is prevented from exceeding a critical level.

> For example in class we have seen the **knee-jerk reflex**. When the patellar tendon below the knee is tapped, the quadriceps muscle is stretched. Sensory neurons detect this stretch and carry the signal into the spinal cord. Inside the spinal cord, the sensory neuron connects directly to motor neurons that control the quadriceps muscle, exciting them. At the same time, the sensory neuron also activates an inhibitory interneuron, which then inhibits the motor neurons that control the opposing hamstring muscle. This dual action (exciting the quadriceps while inhibiting the hamstring) is called reciprocal inhibition, and it ensures smooth, coordinated movement. As a result, the quadriceps contracts, the hamstring relaxes, and the leg extends in a quick, involuntary kick.

## 2.9 Discuss the organization and functions of the peripheral nervous system (PNS), including the somatic and autonomic divisions. Explain how the sympathetic and parasympathetic systems operate antagonistically to maintain homeostasis.

The PNS is responsible to connect the CNS to all the body components, providing it with inputs and then carrying outputs. It can be divided initially in:

- Somatic: receive information from the skin, muscles, and joints. Its receptors transduce physicial energy from them into electric signals.
- Autonomic: it receive information of visceral sensation as well as motorcontrol of the viscera, vascular system, and exocrine glands. It is divided into 2 antagonistic systems that operate on the same body paryts:
  - Sympathetic (fight-or-flight): it uses norepinephrine. It prepares the body for action by stimulating the production of adrenaline. It diverts blood to the somatic musculature.
  - Parasympathetic (rest-and-digest): it uses acetylcholine. It helps the body with restoring and maintaining itself. It diverts blood to the digestive tract.

Together, they ensure homeostasis by modulating organ function in a coordinated, complementary manner, enabling the body to respond appropriately to challenges while supporting long-term health and restoration.

## 2.10 Discuss the organization (direct and indirect pathways) and role of the basal ganglia in motor control, reinforcement learning, and goal-oriented behavior.

The basal ganglia are a group of subcortical nuclei located deep within the cerebral hemispheres. They play a central role in motor control, action selection, reinforcement learning, and goal-directed behavior. Their function is not to initiate movement directly, but to facilitate desired movements and inhibit undesired ones through a system of parallel pathways.

The procedure is as follows:

1. The cortex sends excitatory signals to the striatum, proposing a potential movement or action.
2. The striatum is the input hub of the basal ganglia. It receives cortical input and divides the signal into two pathways:

   - direct, which promotes movement,
   - indirect, which inhibits movement.

   Dopamine excites the direct pathway and inhibits the indirect pathway and it's released proportionally to the error between predicted future reward and actual reward.

   The switch between the two pathways is done thanks to substantia nigra pars compacta that releases dopamine to modulate the plasticity of the striatum, so as to reinforce rewarded actions and make them more likely to recur.
3. The thalamus receives the final inhibitory signal from GPi/SNr. If the direct pathway is active then thalamus is disinhibited and so sends strong excitatory feedback to the motor cortex. Otherwise sends a weak feedback to cortex.

## 2.11 Describe the organization of the cerebral cortex, including its lobes and their key functions.

The cerebral cortex is the outer layer of the cerebral hemispheres, composed of gray matter. It is highly folded, which increase its surface area while allowing neurons to be arranged in closer three-dimensional proximity. This folding reduces axonal length and speeds neural conduction between cortical regions.

The cortex is anatomically organized into two symmetrical hemispheres connected by the corpus callosum, and divided into four main lobes, each associated with distinct functions:

1. **Frontal Lobe**:

   involved in voluntary motor control, executive functions (planning, decision-making, goal-directed behavior), and speech production.
2. **Parietal Lobe**:

   processes somatosensory input (touch, pain, temperature) and supports spatial awareness, navigation, and multisensory integration, critical for bodily and environmental representation.
3. **Temporal Lobe**:

   key for auditory processing, language comprehension, memory formation (especially declarative memory), and visual recognition.
4. **Occipital Lobe**:

   dedicated to visual processing. Receives and interprets visual information from the eyes, enabling perception of shape, color, motion, and depth.

# 3 Introduction to animal reinforcement learning

## 3.1 Discuss the challenges associated with optimal decision-making, including delayed outcomes and the credit assignment problem. How do multiple learning systems in animals address these challenges

Making the best choice is tricky for two main reasons:

1. Delayed Outcomes: the good or bad result of an action often comes later, not right away. It's hard to link a current action to a future reward or punishment.
2. The Credit Assignment Problem: when a good outcome finally happens after a long series of actions, which specific action in that chain actually caused it? It's a hard problem.

In order to address those problems the brain doesn't use one general-purpose system. Instead, it uses three specialized systems that work together:

1. Pavlovian: it learns what events or cues in the world predict good or bad things. It solves the delay problem by letting the animal prepare in advance.
2. Habitual: it turns sequences of actions that worked in the past into automatic routines. Once learned, you do them without thinking. In this way we don't have to recalculate them frequently, which is very efficient.
3. Goal-Directed: it constantly evaluates actions based on their specific outcome to build a plan and can adjust it immediately if the value of an outcome changes. It is the main system for solving the credit assignment problem. Because it tracks the expected result of each action, it can figure out which action in a chain should be credited or blamed when something changes. It handles new or changing situations.

Those three systems work together.

- All new actions are handled by the goal-directed systems and only after multiple positive outcomes during time  they are considered reliable.
- All reliable and predicted sequence of actions in handled automatically by habitual system.
- The pavlovian system works in background providing fast warnings and prep signals to the other two systems

## 3.2 Compare and contrast Pavlovian and instrumental learning, providing examples of how each type of learning influences behavior in everyday life.

Pavlovian Learning and Instrumental Learning are two fundamental ways in which animals learn from experience, but they operate in different ways and serve different functions.

- **Pavlovian learning (two stimuli)**

  > Pavlovian learning is the process by which an organism learns an association between a biologically neutral Conditioned Stimulus (CS) and a biologically significant Unconditioned Stimulus (US), resulting in a Conditioned Response (CR).
  >

  The learning process is:

  1. You sense a stimulus that means nothing to you
  2. Then that stimulus is paired with a meaningful stimulus, that can be positive or negative.
  3. Now the originally neutral stimulus starts to trigger a similar response on its own. Meaning that we are learning to predict the future.

  It crucially depends on structures like the amygdala (necessary for the physiological/emotional CR) and involves Hebbian plasticity, where coincident activation of CS and US pathways strengthens synaptic connections. The hippocampus is involved in the declarative knowledge of the CS-US contingency.

  Here the learner is passive and reactive.

  > **Everyday example**: when you smell food cooking (neutral stimulus) just before eating dinner (meaningful event), the smell alone can eventually make your mouth water, even if you aren’t about to eat yet.
  >
- **Instrumental learning (action and outcome)**

  > Instrumental learning is the process by which an organism learns an association between a voluntary action and its consequential outcome, thereby increasing or decreasing the future probability of that action.
  >

  Here the learner is active and operant.
  The consequences for the actions can be considered as positive/negative reinforcement/punishment.
  Moreover the frequency of the outcome plays a crucial role:

  1. Fixed-ratio: reinforcement is delivered only after a fixed, specific number of responses. It learns that effort translates directly and predictably into reward.
  2. Variable-ratio: reinforcement is delivered after an unpredictable, varying number of responses. This schedule generates the highest and most consistent rate of responding, with virtually no pauses, and creates behavior that is extremely resistant to extinction. This is because the learner anticipates that the next response might be the one that pays off.
  3. Fixed-interval: reinforcement becomes available only after a fixed, predictable amount of time has passed since the last reward. It learns that time, not effort, is the critical factor. So the result is that we see very few responses immediately after a reward, followed by a gradually accelerating rate as the time for the next reward approaches.
  4. Variable-interval: reinforcement becomes available after unpredictable, varying intervals of time. This schedule produces a slow, steady, and persistent rate of response, as the learner cannot predict when the opportunity for reward will next be available but knows that consistent checking will eventually pay off.

  > **Everyday example**: checking a smartphone for social media likes. The delivery of likes is on a variable-ratio schedule (unpredictable number of checks required). This schedule produces a high, steady rate of checking behavior that is highly resistant to extinction.
  >

## 3.3 Discuss how the processes of acquisition, extinction, generalization, and discrimination in Pavlovian conditioning demonstrate the flexibility of learning.

The processes of acquisition, extinction, generalization, and discrimination in Pavlovian conditioning collectively illustrate that learning is not a rigid, all-or-nothing imprint, but a dynamic, flexible, and context-sensitive system that allows an organism to adaptively update its predictions about the world.

1. **Acquisition**

   It is the initial learning phase where a neutral Conditioned Stimulus (CS) is repeatedly paired with a biologically significant Unconditioned Stimulus (US), leading to the emergence of a Conditioned Response (CR). The curve of strenght growth over time is exponential, as it is important to quickly learn something that is meaningful to survival.
2. **Extinction**

   It occurs when the CS is repeatedly presented without the US, leading to a decrease in the CR. The curve of strenght decrease over time is exponential. It is important to change the behaviour quickly when it is not meaningful in order to quicklier adapt to the current enviroment and so to survive.

   It is important to highlight that extinction is not the same as forgetting but rather suppressing a response. In support of that many experiments showed some interesting features of extinction:

   - **Spontaneous recovery**: after a time delay following extinction, the CR can reappear.
   - **Renewal** effect: if acquisition occurs in Context A and extinction in Context B, the CR returns when the CS is presented again in Context A. This means that the processes are context-specific
   - **Reinstatement**: presenting the US alone after extinction can cause the CR to the CS to return. This shows the status of the US can retroactively update the strength of the learned association.
3. **Generalization**

   It is when stimuli that are similar to the original CS also come to elicit a CR.

   This is a very powerful mechanism as it allows for efficient and rapid responding to novel situations without requiring direct experience with every possible stimulus. It can be adaptive or maladaptive, leading to inappropriate fear responses to safe stimuli for example.
4. **Discrimination**

   It is the learned ability to differentiate between a CS that predicts the US and a similar stimulus that does not.

   This process demonstrates inhibitory learning. The organism learns not only when something will happen but also when it will not, which is equally critical for efficient resource allocation and avoiding unnecessary responses.

## 3.4 How do conditioned responses, which are initiated in anticipation of the unconditioned stimulus, highlight the predictive nature of learning, and why is this predictive capacity advantageous for survival?

The key is to understand that a Conditioned Response (CR) is not a reaction to the thing that matters (the Unconditioned Stimulus or US, like food or pain). Instead, it is a reaction to the signal that predicts it (the Conditioned Stimulus or CS, like a bell or a light). The simple fact that the response happens before the important event occurs is the proof that learning is predictive.

Reacting to dangers or opportunities after they happen is often too late. Prediction allows you to get ready beforehand, which offers major advantages:

- **Better Protection**: a blink that finishes as the puff hits is useless. A blink timed to be fully closed before the puff hits actually protects the eye. Similarly, freezing at the sound of a predator's rustle (CS) before it sees you is better than freezing after it attacks (US).
- **Improved Preparation**: predictive learning lets your body prepare systems in advance. Salivating (CR) at the smell of food (CS) primes your digestive system for better nutrient absorption once the food (US) arrives.
- **Energy Efficiency**: it allows for smarter resource use. Instead of being in a constant state of high alert, an animal can use predictive cues to turn on its stress or preparation systems only when they are likely to be needed. This conserves vital energy.
- **Enables Planning**: a purely reactive creature can only respond. A predictive creature can plan. For example, if a certain cloud formation (CS) predicts rain (US), the prediction allows you to take the instrumental action of seeking shelter in advance. Prediction is the first, essential step towards goal-directed behavior.

## 3.5 Explain how the principles of neural plasticity underlie learning and memory. Discuss the role of synaptic changes and provide specific examples, such as the gill withdrawal reflex in Aplysia.

Learning and memory are not abstract processes but are physically embodied in the brain through the fundamental principle of neural plasticity. Plasticity refers to the ability of the brain's neural circuits to change their structure and function in response to experience. This enduring alteration in synaptic efficacy and connectivity is the primary biological mechanism underlying the acquisition, storage, and retrieval of memories.

At the core of this process are synaptic changes, modifications in the strength and number of connections between neurons. These changes can be broadly categorized:

- **Short-term plasticity**: involves transient, functional changes in the effectiveness of existing synapses, often through biochemical modifications that enhance or depress neurotransmitter release. This is closely associated with Hebbian plasticity, summarized by the axiom "neurons that fire together, wire together." When a presynaptic neuron consistently and persistently activates a postsynaptic neuron, the synaptic connection between them is strengthened. This provides a cellular mechanism for forming associations, such as those between a conditioned and unconditioned stimulus.
- **Long-term plasticity**: involves more permanent structural changes, including the growth of new synaptic connections, the pruning of unused ones, and the anatomical remodeling of neural networks. These changes, which can last a lifetime, are the basis for long-term memory consolidation.

**Example: The gill withdrawal reflex in Aplysia Californica**

- The Basic Circuit: A weak tactile stimulus to the snail's siphon (the CS pathway) triggers a mild, reflexive gill withdrawal via a direct sensory-to-motor neuron synapse.
- Short-Term Facilitation (Learning): When the siphon touch (CS) is paired with a strong, noxious shock to the tail (US), the shock activates modulatory interneurons. These interneurons release serotonin onto the synapse between the siphon sensory neuron and the gill motor neuron. This serotonin release initiates a biochemical cascade within the sensory neuron that results in enhanced neurotransmitter release when the siphon is subsequently touched. Consequently, the same weak touch now produces a much stronger and longer-lasting gill withdrawal. This is a direct manifestation of Hebbian plasticity: the near-simultaneous activation of the CS (siphon touch) and US (tail shock) pathways strengthens the specific synapse linking them.
- Long-Term Memory (Structural Plasticity): Repeated pairings of the CS and US trigger a more profound change. The biochemical signals shift from merely modifying proteins to activating genes that promote structural growth. The sensory neuron grows new synaptic terminals onto the motor neuron, effectively creating more connections at the same junction. This anatomical change underlies the long-term memory of the association. After conditioning, a touch to the siphon alone can elicit a robust withdrawal response that lasts for days or weeks.

## 3.6 Describe the process of extinction in Pavlovian conditioning. What evidence suggests that extinction is a form of new learning rather than simply forgetting? Discuss the implications of extinction for therapeutic interventions targeting maladaptive behaviors.

See the [answer 3.3](#3-3).

The modern understanding of extinction forms the core rationale for exposure-based therapies, which are the gold-standard treatment for anxiety disorders (e.g., phobias, PTSD, OCD), addictions, and other conditions driven by maladaptive learned associations.

> Therapy is not about "erasing" a traumatic memory or a fear of spiders. Rather, it is about creating a new, powerful, and safe memory that can override the old, maladaptive one. The therapeutic session is a controlled "extinction trial."

As we have seen extinction is dynamic and a learned CR is not so easy to override:

1. **Contextual Variety** (Combating Renewal): exposure is conducted in multiple contexts to make the extinction memory more generalizable and less tied to the "therapy room" context. This helps prevent relapse when the patient encounters the feared stimulus in a new environment.
2. **Stress Management** (Combating Reinstatement): since stress can function like an unsignaled US and trigger reinstatement of fear, therapy often incorporates stress-management techniques. Furthermore, exposure is conducted in a way that ensures the patient's anxiety (the CR) decreases within the session and across sessions, solidifying the new safety memory.
3. **Booster Sessions and Spaced Practice** (Combating Spontaneous Recovery): therapy is structured over multiple, spaced sessions rather than one long marathon. This repeated reactivation and re-extinction of the memory strengthen the inhibitory learning. Follow-up "booster" sessions are used to counteract the natural passage of time, which can allow for spontaneous recovery of the old memory.

For new inhibitory learning to occur, the brain must experience a violation of expectation. The patient must fully encounter the CS and learn that the expected catastrophic US does not occur.

## 3.7 Discuss the evidence related to how a lesion to the amygdala or hippocampus differently impair Pavlovian conditioning.

In class we have seen a study that provided causal evidence for a "double dissociation", showing that these two medial temporal lobe structures are necessary for fundamentally different aspects of the learning process.

> **Experiment**: the study involved two parallel Pavlovian conditioning experiments. In both, an aversive unconditioned stimulus (US), a loud sound, was used. The experiments differed only in the modality of the conditioned stimulus (CS): one used a visual CS (a colored slide), the other an auditory CS (a specific tone). The critical participants were three patients with selective, bilateral brain damage: one with damage confined to the amygdala, one with damage confined to the hippocampal formation, and one with damage to both structures. Their performance was compared to healthy controls.

Two key dependent measures were tracked simultaneously:

1. The psychophysiological conditioned response (CR), measured as the skin conductance response, an index of autonomic arousal and emotional learning.
2. Declarative knowledge of the CS-US contingency, measured by verbally questioning participants about which stimulus predicted the noise.

The findings revealed a clear and complementary pattern of impairment.

- The patient with selective amygdala damage exhibited a profound deficit in generating the autonomic conditioned response to the CS. Crucially, she performed normally on the declarative test, she could verbally report and explain which stimulus predicted the loud noise. This shows that the amygdala is necessary for the implicit, emotional, and physiological expression of Pavlovian conditioning, but not for the conscious, declarative knowledge of the association.
- Conversely, the patient with selective hippocampal damage showed the opposite profile. He developed a normal skin conductance response to the CS, indistinguishable from controls. However, he was completely impaired on the declarative knowledge test, he could not state which stimulus was paired with the noise, despite his body demonstrating a clear learned response. This demonstrates that the hippocampus is necessary for the explicit, declarative memory of the CS-US relationship, but not for the acquisition of the implicit conditioned emotional response.
- The patient with combined damage was impaired on both measures, confirming that the two processes are functionally separate and rely on distinct neural substrates.

> This double dissociation provides definitive evidence that Pavlovian conditioning is not a unitary process. It involves at least two parallel learning systems. An **amygdala-dependent** system that underlies the acquisition and expression of implicit emotional memories, physiological responses, and stimulus-value associations. A **hippocampus-dependent** system that underlies the formation of explicit, declarative memories about the relationships between events in the world.

## 3.8 Explain the different reinforcement schedules (fixed-ratio, variable-ratio, fixed-interval, variable-interval) and their effects on behavior in instrumental learning. Provide real-world examples of each schedule and discuss their implications for shaping and maintaining behavior.

For the explanation of schedules see [answer 3.2](#3-2).

For what concerns the examples:

1. Fixed-ratio: a video game where a player must defeat exactly 50 enemies to earn a new weapon or achievement. The player learns that a specific, predictable amount of effort always yields the reward.
2. Variable-ratio: gambling on a slot machine or buying lottery tickets. The reward (a payout) occurs after an unknown number of pulls or purchases, which powerfully reinforces persistent engagement.
3. Fixed-interval: receiving a weekly paycheck or checking an oven timer as the finish time approaches. The behavior (work, checking) increases as the known time of reward delivery nears.
4. Variable-interval: checking email or a messaging app for responses. Because replies arrive at irregular intervals, this schedule encourages slow, steady, and persistent checking throughout the day.

For what concerns implications for shaping and maintaining behavior:

1. Acquisition vs. Persistence: a Fixed-Ratio schedule is effective for quickly establishing a new behavior by making the reward contingencies clear. However, to make a behavior highly persistent and resistant to extinction a Variable-Ratio schedule is far more powerful. Its unpredictability means the learner continues responding long after rewards have stopped, hoping the next action will pay off.
2. Efficiency and Pattern of Response: fixed schedules (FR/FI) tend to produce a "stop-start" pattern of behavior. This can be inefficient for maintaining steady performance. In contrast, Variable schedules (VR/VI) generate more consistent, steady rates of responding, which is useful for maintaining engagement in activities like customer loyalty programs (VR) or consistent vigilance (VI).

# 4 Contiguity, Contingency and Surprise as drivers of Reinforcement Learning

## 4.1 Critically evaluate the historical shift in understanding the conditions necessary for learning, from an emphasis on contiguity to the recognition of the importance of contingency and surprise. Use specific experimental examples to support your arguments.

1. The initial, contiguity-based view of learning held that the simple temporal closeness of a conditioned stimulus (CS) and an unconditioned stimulus (US) was both necessary and sufficient for an association to form. This principle was tested and seemingly supported by experiments comparing two fundamental conditioning procedures: delay conditioning and trace conditioning.

   - In **delay conditioning**, the CS is presented and continues until the US begins, so the two stimuli overlap in time. This creates maximal temporal contiguity.
   - In contrast, in **trace conditioning**, the CS is presented and then terminated, leaving a silent interval—the trace interval—before the US occurs. This procedure reduces the immediacy of contiguity.

   Early experiments consistently demonstrated that conditioned responses are acquired more readily and strongly under delay conditioning than under trace conditioning. For instance, studies with rats showed that with a short interval between a tone CS and a food US, robust anticipatory head entries developed. When the interval was lengthened, the conditioned response was significantly weaker and slower to emerge. These findings were interpreted as direct evidence for the contiguity principle: learning suffers when the temporal gap between CS and US is increased, presumably because the neural trace of the CS fades before the US arrives to form an association. Thus, delay conditioning, with its perfect contiguity, served as the paradigm case supporting the idea that "closeness in time" is the critical determinant of learning.
2. However, this view was challenged and refined through a series of insightful experiments that demonstrated contiguity alone is insufficient. For example in an experiment, rats received tones (CS) and shocks (US) randomly and independently, such that the probability of shock was equal in the presence and absence of the tone. More precisely there were three groups:

   1. Random: US and CS were programmed independently and randomly throughout each session. Critically, the shocks were "equiprobable at any time within the session," meaning they were just as likely to occur during the tone as in its absence.
   2. Gated: same number of CS-US pairings as previous group, but unpaired USs were removed. US occurrence is gated to CS occurrence. No US occurring in absence of CS. Here, the tone was a perfect predictor of the shock.
   3. Random-Gated: same number of US as in group G but random occurrance. This group served as an additional control. This controlled for the possibility that the sheer density or total number of US presentations, rather than their contingency with the CS, could affect learning.

   Only the group G showed conditioning. This elegantly demonstrated that learning depends on contingency, the predictive relationship where the probability of the US given the CS is greater than the probability of the US in its absence. Formally $p(US|CS)>p(US|\neg CS)$. Organisms do not merely associate contiguous events, they learn about the informational or causal structure of their environment.
3. The understanding deepened further with the discovery of the blocking effect. In a typical blocking experiment, an animal first learns that a tone (CS1) predicts a US. In a second phase, a light (CS2) is presented together with the tone, followed by the same US. Even though the light is contiguous with the US, it fails to elicit a conditioned response. This occurs because the US is already fully predicted by the tone. The light provides no new information and thus no surprise. Blocking showed that contiguity and contingency are still not enough. Learning only happens when there is a prediction error, a discrepancy between what is expected and what actually occurs. Surprise, therefore, acts as a gatekeeper for learning: organisms update their expectations only when outcomes violate predictions.

## 4.2 Discuss the role of prediction error in reinforcement learning. How is this concept formalized in computational models like the Rescorla-Wagner model, and what are the implications for understanding how organisms learn from their experiences?

Prediction error is the fundamental computational concept at the heart of modern reinforcement learning theory. It serves as the essential teaching signal that drives the updating of an organism's internal model of the world. Conceptually, it is the discrepancy between what is expected and what actually occurs. This error signal dictates both the direction and magnitude of learning: when outcomes are perfectly predicted, no error exists and no new learning takes place. It is only when predictions are violated, when we are surprised, that our expectations are revised.

The **Rescorla-Wagner model** proposes that on any given trial, the change in the associative strength (V) of a conditioned stimulus (CS) is proportional to the prediction error experienced on that trial. Where the prediction error at the time $t$ is $\delta_t = R_t - V_t$, while the next associative strenght will be $V_{t+1}=V_t + \alpha\delta_t$. With this differential equations it was possible to infer the acquisition and extintion curves. It also models the blocking-effect as if the error is 0 then $V_{t+1}=V_t$.

Anyway this model has some limitations:

1. It deals with how associative strengths change from trial to trial without considering any details about what happens within and between trials. We consider snapshots of time and not a real flow of time. This means it cannot explain how organisms learn precise temporal relationships.
2. The model fails to account for second-order conditioning. In this phenomenon, a neutral stimulus (CS2) is paired not with a primary US, but with an already-conditioned stimulus (CS1). The mdoel predicts a negative error as the US does not occur and should lead to extinction while in practice we have acquisition.

## 4.3 Compare and contrast the Rescorla-Wagner and Temporal Difference models of learning. In what ways does the TD model offer a more nuanced account of learning, particularly with respect to the temporal aspects of conditioning?

Details on Rescorla-Wagner are on the [previous answer](#4-2).

The core contrast lies in their treatment of time: where Rescorla-Wagner is a trial-level model, Temporal Difference (TD) is a real-time, within-trial model. It breaks the continuous stream of experience into a series of discrete time steps. Its central innovation is to define the prediction error not as the difference between an outcome and a prediction, but as the difference between successive predictions of total future reward, plus any immediate reward received. The associative strenght is updated as before $V_{t+1}=V_t + \alpha\delta_t$ while the prediction error now is $\delta_t=R_t+(V_t-V_{t-1})$. In simpler terms, the error is triggered whenever the aggregate prediction of the future changes from one moment to the next.

In this way this new model can addess some previous issues:

1. **Learning the Timing of Events**: the TD model can explain how animals learn when a US will occur, not just that it will occur. As learning progresses, the positive prediction error generated at the moment of US delivery propagates backwards in time, step-by-step, strengthening predictions at earlier and earlier time steps within the CS.
2. **Explaining Second-Order Conditioning**: This phenomenon, problematic for RW, is naturally explained by TD. When CS2 (a light) is paired with an already-valued CS1 (a tone), the onset of CS1 acts as an "internal reward" because it predicts the future primary US. The TD error signal is therefore positive at the moment CS1 occurs in the presence of CS2. This error reinforces the association for CS2, allowing it to acquire value by predicting another predictive stimulus, effectively chaining predictions together over time.

## 4.4 Explain the blocking effect and its significance for our understanding of reinforcement learning. How does this phenomenon demonstrate that learning is not simply about forming associations between contiguous stimuli? What does it tell us about the role of prediction and surprise?

The question was answered in [question 4.1](#4-1)

## 4.5 Discuss the broader implications of predictive learning beyond classical conditioning, particularly in the domain of sensory perception. Provide examples of how the brain actively constructs our experience by generating and updating predictions about the sensory world.

The principles of predictive learning, first formalized in the context of classical conditioning, extend far beyond the domain of simple associations to provide a foundational framework for understanding cognition itself, most profoundly in the realm of sensory perception. This broader implication is captured by the predictive processing theory, which posits that the brain is not a passive receiver of sensory data but an active, generative model that constructs our perceptual experience through a continuous cycle of prediction and prediction-error minimization.

> A checkerboard is shown with a green cylinder casting a soft shadow across it. Two specific squares are indicated: Square A, which is a dark square physically located in the lit area, and Square B, which is a light square physically located within the shadow. To our local photoreceptors and a simple light meter, Square A reflects more light (luminance) into the eye than Square B. Yet, we perceive Square A as a dark grey and Square B as a light white. Most astonishingly, when the surrounding context is removed and only the two squares are shown, they are revealed to be physically identical in shade.

This phenomenon is not a flaw in our vision but a supreme feature of our predictive perceptual system. The brain does not passively register luminance. It actively infers the most likely reflectance by discounting the inferred lighting conditions.

# 5 The reward prediction error hypothesis of dopamine neurons

## 5.1 Discuss the experimental evidence that supports the reward prediction error hypothesis of dopamine neuron activity.

The experiment seen in class had the following setup. An alert monkey is seated with electrodes implanted to record the activity of single dopamine neurons in the midbrain. The monkey receives a drop of fruit juice as a reward. A neutral conditioned stimulus (CS), such as a light or tone, is introduced as a predictor.

The various phases of the experiment were:

1. Before learning

   The reward is given unexpectedly, so there is no association between CS and reward. Each time the reward is delivered we can observe a spike in phasic response of dopaminergic neurons. This happens beacuse the reward is always unexpected so there is always a large and positive prediction error.
2. Acquisition

   The CS is repeatedly paired with the reward. The animal is supposed to learn the association. We observe so that phasic response gradually shifts from the time of reward delivery to the onset of the CS. Moreover the response to the actual reward diminishes as it becomes fully predicted. This follows as the reward now is fully predicted when the CS occur, so there is no prediction error and so no spike on the reward. The spike shifts to the CS as it is not predictable.
3. Manipulations

   Then the scientists applied some modifications to inspect the response of dopaminergic activity and infer some interesting features:

   - **Unexpected Reward Omission**: the CS is presented but without reward. A phasic burst occurs at the CS, but at the precise time the reward was expected, firing rates drop below baseline. This can be seen as a negative prediction error. Moreover this suggests also a time prediction since the dopaminergic drop occurs when the reward should have been provided.
   - **Relative Value Encoding**: the same physical reward can either excite or inhibit dopamine neurons depending on what was expected. If a small reward was predicted, a medium reward excites. If a large reward was predicted, the same medium reward inhibits. This reveals that dopamine encode relative surprise and not absolute one.
   - **Probability/Uncertainty Encoding**: the phasic response to a CS is largest when it predicts a reward with an intermediate probability (50%), not with certainty (100%). The dopamine signal scales with the degree of uncertainty or the "new value" of the predictor, maximizing when the outcome is most informative for learning.

## 5.2 Explain the role of dopamine and the basal ganglia in reinforcement learning and goal-oriented behavior. How do the direct and indirect pathways contribute to this process?

See [answer 2.10](#2-10)

Moreover we have seen an experiment on the causal role of dopamine in driving reinforcement learning and choice behavior. In order to inspect the changes groups of healthy participants were establoshed. Then each group was given a grug to alter dopamine function: an antagonist,  a precursor and a placebo.

Then the participants have to chose between visual stimuli that could lead them to a gain, loss or neutral condition.

The participant with enhanced dopaminergic function had higher selection of high-reward stimuli than neutral. On the other hand those with reduced dopaminergic function were less likely to choose those stimuli. It is interesting to notice that those differences did not take place in negative-reward stimuli.

## 5.3 Critically evaluate the statement: "Dopamine is the 'feel-good' neurotransmitter." How does the reward prediction error hypothesis challeng2e this view, and what are the implications for understanding reward processing?

The "feel-good" label stems from observations linking dopamine to pleasure and euphoria. For instance, addictive drugs that produce euphoria often increase dopamine levels, and electrical stimulation of dopamine pathways can be intensely rewarding. However, this view is limited and misleading in some ways:

1. Temporal Disconnect: dopamine releases, especially the phasic bursts that signal prediction errors, occur on a millisecond scale, while subjective feelings of pleasure unfold over seconds. They are not temporally aligned.
2. Dopamine in Aversive Contexts: dopamine neurons can also be activated by salient, surprising, or aversive stimuli, not solely by pleasurable ones.

The RPE hypothesis (see [answer 5.1](#5-1)):

1. Dopamine is now understood as crucial for motivation, incentive salience, and goal-directed pursuit ("wanting"). This is separable from the actual experience of pleasure ("liking"). This distinction clarifies conditions like addiction, where dopamine-driven "wanting" becomes pathologically strong even as the "liking" for the drug diminishes.
2. The negative dip in dopamine activity for negative prediction errors is an active teaching signal that discourages unrewarded behaviors. Thus, dopamine is involved in learning from both positive and negative outcomes.
3. Addiction can be seen as a hijacking of the RPE system. Drugs of abuse often cause massive, pharmacologically forced dopamine surges that are not modulated by expectation. The brain interprets these as enormous, perpetual prediction errors, irrationally inflating the value of drug-associated cues and driving compulsive seeking

## 5.4 Describe how the reward prediction error framework can be applied to understand the neural mechanisms underlying drug addiction. What are the key differences in dopamine signaling between natural rewards and addictive drugs?

In healthy reinforcement learning, dopamine neurons release phasic bursts that encode prediction errors, teaching signals that update the value of cues and actions. This system ensures behavior remains adaptive: unexpected rewards strengthen associations, while fully predicted rewards no longer elicit dopamine, and omitted rewards cause dips that weaken associations.

> Addictive drugs hijack this precise teaching mechanism. They do not simply stimulate the reward system, they corrupt its computational logic by generating dopamine signals that are pharmacologically imposed and decoupled from prediction.

1. Persistent, Unmodulated Dopamine Surges: unlike natural rewards, drugs of abuse cause large, direct dopamine releases in the striatum via their pharmacological action on transporters or neurons. Critically, these surges do not diminish with learning, they occur regardless of expectation. Every drug dose acts like an enormous, unpredictable reward, generating a perpetual “positive prediction error” signal.
2. Maladaptive Learning and Inflated Cue Values: according to the RPE framework, a persistent positive prediction error will continuously increase the learned value of associated cues and contexts. Thus, drug-associated cues become hypervalent—assigned an ever-increasing motivational significance that far exceeds their true utility.
3. Blunting of Natural Reward Responses: chronic drug exposure leads to neuroadaptive changes, including downregulation of dopamine receptors and reduced baseline dopamine function. This impairs the system’s ability to respond to natural rewards, which now generate smaller prediction errors in comparison. Consequently, natural rewards lose their motivational power—a phenomenon observed as anhedonia in addiction.
4. Transition to Habitual and Compulsive Use: the constant, prediction-error-like dopamine bursts reinforce drug-taking actions directly and inflexibly. Over time, control over behavior shifts from goal-directed circuits to habitual striatal circuits. Behavior becomes compulsive because the dopamine signal falsely indicates that taking the drug is always better than expected.

## 5.5 Discuss the relationship between prediction errors, synaptic plasticity, and learning. How does dopamine contribute to these processes at a neural circuit level?

At a computational level, prediction errors, discrepancies between expected and actual outcomes, serve as teaching signals. They indicate when the brain’s model of the world is inaccurate and needs revision. For learning to occur, these error signals must be translated into durable changes in neural connectivity that favor successful predictions and actions while discouraging unsuccessful ones. This physical change is achieved through synaptic plasticity, the activity-dependent strengthening or weakening of synaptic connections between neurons.

When a particular pattern of neural activity consistently leads to a positive outcome, the synapses involved in that pattern should be strengthened. Conversely, synapses supporting patterns that lead to negative outcomes or violated predictions should be weakened. However, for this process to be guided by experience, it requires a neuromodulatory signal that carries information about the outcome’s value and surprise. This is where dopamine enters the circuit.

Dopamine modulates how other neurons communicate and change. By broadcasting a phasic reward prediction error signal, dopamine tells widespread brain regions when and where synaptic changes should occur. For example see the [basal ganglia motor loop](#5-2).

# 6 From reinforcement learning to decision-making: goals and habits in the brain

## 6.1 Discuss the historical progression of thought regarding learning and behavioral control, tracing the evolution from stimulus-response theories to the understanding of goal-directed and habitual actions as distinct but interacting systems.

This evolution can be articulated through four distinct but interconnected generations of research, each building upon and challenging the previous paradigm.

1. **Generation-0**:

   > Is learning merely the formation of direct stimulus-response (S-R) bonds, or does it involve the creation of internal mental representations of the world?
   >

   - The first belived that learning is the strengthening or weakening of S-R connections based solely on reinforcement. The organism is a passive responder to environmental cues.
   - The second beleived that learning can occur without immediate reinforcement, and this latent knowledge is used flexibly when a goal becomes relevant.

   In order to test this debate an experiment was made:

   1. Rats were placed in a complex maze with many blind alleys. The maze featured doors and curtains to prevent visual long-range planning, forcing rats to explore sequentially.
   2. Some rats were given no reward for solving the maze, others were given a consistent reward and the last group was given no reward at the beginning but then it was introduced.
   3. As results, the group 2 learned the maze fastest. Critically, group 3 showed an immediate and dramatic improvement in performance once they ere given reward, matching Group 2's performance almost at once.

   The rats in roup 3 had learned the maze's layout ("cognitive map") during the unrewarded exploration. This latent learning remained hidden until motivation (hunger + a known food reward) was provided. This experiment directly challenged S-R theory and provided evidence for internal representation.
2. **Generation-1**

   > This generation moved from spatial maps to a general theory of action control.
   >

   An action can be made following two type of behaviours:

   - Goal-directed: an action chosen because of a known relationship between the action and a desired outcome. It is flexible, deliberative, and sensitive to changes in the outcome's value or the action-outcome contingency.
   - Habitual: an action triggered automatically by a stimulus based on past reinforcement history. It is efficient, inflexible, and persists even if the outcome is no longer desired or the action no longer produces it.

   In order to dissociate the two systems we can apply a procedure:

   1. Training: an animal learns that a specific action leads to a specific outcome.
   2. Post-training Manipulation: we can change the desirability of the action in two ways by reinforcer devaluation, provoking satiety or taste aversion, or by contingency degradation, providing the reward randomly.
   3. Testing: if the action is less performed then it is goal-directed othwerwise it is habitual.

   > **NB**: Early in training, behavior is goal-directed. After extensive overtraining, it becomes habitual.
   >

   This dissociation has neural representations:

   1. Dorsomedial striatum $\rightarrow$ goal-directed
   2. Dorsolateral striatum $\rightarrow$ habitual

   They are activate in parallel during the basal ganglia motor loop and concur to the final decision on which pathway to take for each proposal.
3. **Generation-2**:

   > Tries to prove the results obtained on animals in the previous generation, but in humans.
   >

   Two different experiments tried to inspect the two approaches:

   1. Goal-oriented

      - Method: participants learned that two different actions led to two different food rewards. One food was then devalued by feeding the participant to satiety. In a subsequent test, brain activity was measured while they chose actions.
      - Result: behaviorally, participants chose the action leading to the devalued food less often. Neurally, the medial Orbitofrontal Cortex (OFC) showed differential activity, tracking the current value of the specific expected outcome. This identified the human OFC as a key node for goal-directed, outcome-value-based choice.
   2. Habitual

      - Method: two groups were trained on a simple button-pressing task for food rewards. The "Habit Group" received extensive training, the "Goal-Directed Group" received minimal training. One food reward was then devalued by satiation.
      - Result: in the test, the 1-day group reduced responses for the devalued outcome (goal-directed). The 3-day habit group continued to press for the devalued outcome. fMRI revealed that habit formation correlated with increased engagement of the posterior dorsolateral striatum (putamen/globus pallidus) in humans.
4. **Generation-3**:

   This generation provided a formal, mathematical language to describe the mechanisms of Generation 1 & 2.

   - **Model-Based** (Goal-Directed): the agent builds and uses an internal "model" of the environment, a set of knowledge about state transitions and rewards. It can plan by mentally simulating future steps before acting. This is flexible but computationally expensive.
   - **Model-free** (Habitual): the agent does not have a model. It learns simple, cached value estimates for actions (or state-action pairs) through trial and error. It acts based on these past summaries. This is efficient but inflexible.

   In order to test those models we have seen a specific experiment: Sequential two-choice Markov decision tasks.
   The experiment is divided into two phases:

   1. First choice: choose between two options. Each has a high probability (70%) of leading to one of two distinct second-stage states and a low probability (30%) of leading to the other.
   2. Second choice: in the reached state, choose between two other options, each with an independent, slowly changing probability of yielding a monetary reward.

   > To maximize reward, an optimal agent must use the transition structure (the model) from the first stage to interpret second-stage outcomes.
   >

   So the two models will behave differently in this experiment:

   - Model-free: simply repeats the first-stage action if it was rewarded, regardless of the transition type.
   - Model-based: it repeats the action after a rewarded common transition, but often switches after a rewarded rare transition. Why? Because a reward after a rare transition is more likely attributed to the alternative first-stage action (which commonly leads to that rewarding second state).

## 6.2 Critically evaluate the experimental methodologies used in animal and human studies to dissociate goal-directed and habitual behaviors. What are the strengths and limitations of reinforcer devaluation and contingency degradation procedures?

They were described in generation-1 and generation-2 in [answer 6.1](#6-1).

1. Reinforcer Devaluation

   - Strenghts

     - Clear Behavioral Readout: it provides a clean, quantifiable measure—reduction in response rate—that directly tests the motivational basis of an action.
     - Cross-Species Validity: the paradigm translates robustly from rodents to humans (generation-2), allowing for direct comparison of neural substrates.
   - Limitations

     - Confound of General Motivation vs. Specific Value: it can be difficult to completely separate a reduction in the specific value of an outcome from a general reduction in motivation or arousal.
     - Temporal and Associative Spillover: in taste aversion procedures, the illness must be carefully timed to devalue the outcome without creating a direct aversion to the action or context. Poor timing can lead to misinterpretation.
     - Limited to Appetitive Behaviors: It is most straightforward with food or fluid rewards. Devaluing other types of reinforcers (e.g., social rewards, drugs of abuse) is methodologically more complex.
2. Contingency Degradation

   - Strenghts

     - Tests Causal Belief Directly: it specifically probes the agent's understanding of the action-outcome contingency, a core pillar of goal-directed control.
     - Controls for Reward Rate: by matching the rate of reward delivery to the training phase, it controls for simple extinction or changes in reward frequency, isolating the effect of degraded contingency.
   - Limitations

     - Detection and Learning Confound: the subject must first detect that the contingency has changed. A failure to reduce responding could reflect either habitual control or a failure to learn about the new degradation contingency.
     - More Cognitively Demanding: it arguably requires a more sophisticated cognitive comparison between past and present contingency structures than devaluation.
     - Implementation Complexity: designing a truly random, action-independent reward schedule that perfectly matches the previous reward density is technically more challenging than simply removing or devaluing an outcome.

## 6.3 Explore the neural substrates underlying goal-directed and habitual behaviors in both rodent and human brains. How do the findings from animal studies translate to our understanding of human decision-making, and what role do different regions of the striatum and prefrontal cortex play?

The neural substrates of goal-directed and habitual behaviors involve dissociable cortico-striatal circuits conserved across species.

1. Rodent brain:

   - Goal-directed: Depends on the dorsomedial striatum (DMS), which receives input from prefrontal cortex (PFC), orbitofrontal cortex (OFC), and amygdala. This circuit encodes response-outcome (R-O) associations.
   - Habitual: Depends on the dorsolateral striatum (DLS), receiving sensorimotor input. It encodes stimulus-response (S-R) bonds.
2. Human brain (translational findings):

   - Goal-directed: Involves anterior caudate (homolog of DMS) and medial OFC, which tracks specific outcome value.
   - Habitual: Involves posterior putamen/dorsolateral striatum (homolog of DLS), activated after overtraining.

Translation to human decision-making:

- Learning and expertise: Skill acquisition shifts neural control from prefrontal–anterior caudate circuits (goal-directed) to sensorimotor–posterior putamen circuits (habitual), automating performance.
- Willpower failures: Under stress or cognitive load, the resource-intensive PFC/OFC goal system is impaired, allowing the robust habit circuit (posterior putamen) to dominate behavior (e.g., stress eating).
- Addiction: Drugs of abuse cause pathological hyper-strengthening of the habit circuit (posterior putamen) and weaken PFC/OFC goal-directed control, leading to compulsive drug, seeking despite negative consequences, a direct analogue of devaluation-insensitive habitual behavior in animals.

Role of striatum and prefrontal cortex:

- Striatum: Site of competition, anterior caudate/DMS supports flexible actions, posterior putamen/DLS supports fixed habits.
- Prefrontal cortex (OFC, LPFC): Provides value signals (OFC) and top-down control (LPFC) to guide or override striatal selection.

## 6.4 Explain the computational distinction between model-based and model-free reinforcement learning. How do these computational frameworks help us understand the cognitive processes underlying goal-directed and habitual control, and what evidence supports the idea that both systems contribute to behavior?

Most of the answer is in [answer 6.1](#6-1).

Moreover these frameworks clarify cognitive processes:

- They translate behavioral characteristics into algorithmic terms: goal-directed planning becomes model-based lookahead search, while habit automaticity becomes model-free cached-value retrieval.
- They explain why goal-directed actions are sensitive to outcome devaluation and contingency degradation (the model is updated), while habits are not (values are cached and rigid).

And there are multiple evidences that both systems contribute to behaviour like the two-step markov task described in [answer 6.1](#6-1).

## 6.5 Consider the implications of the dual-systems perspective of behavioral control (goal-directed vs. habitual, or model-based vs. model-free) for understanding various aspects of human behavior, such as addiction, learning new skills, and adapting to changing environments.

- Addiction: it reflects a pathological dominance of the habitual/model-free system coupled with impairment of the goal-directed/model-based system. Drug use initially may be goal-directed (seeking euphoria), but with repetition, control shifts to the habit circuit (dorsolateral striatum/posterior putamen). Simultaneously, chronic substance abuse damages prefrontal regions (OFC, medial PFC) responsible for outcome valuation and impulse control. This results in compulsive drug-seeking that is insensitive to devaluation—the individual continues using despite catastrophic consequences. The model-free system’s cached values—super-charged by dopamine surges—override the weakened model-based system’s understanding of long-term harm.
- Learning new skills: skill acquisition illustrates a functional transition from model-based to model-free control. Early learning is slow, effortful, and goal-directed: the learner uses a model-based approach to consciously plan each step. This heavily involves prefrontal cortex and anterior caudate. With practice, performance becomes fluent and automatic, a shift to model-free control. The routine is encoded in the motor loop involving the posterior putamen, freeing cognitive resources for higher-level goals. This transition explains the progression from deliberate practice to effortless expertise.
- Adapting to new enviroment: successful adaptation requires dynamic arbitration between the two systems. In stable environments, habits (model-free) are efficient. When the environment changes, such as a new work procedure or a road closure, the model-based system must detect the change, inhibit the old habit, and plan a new action. Failures to adapt occur when the habitual system is too strong or the goal-directed system is compromised. This explains why people persist with outdated behaviors even when aware of new goals. Effective adaptation thus depends on prefrontal capacity to override the striatal habit loop and reconfigure behavior flexibly.

# 7. Module-2

## 7.1 What is the ventral visual stream, and why is it important for visual object recognition?

The ventral visual stream is the cortical pathway responsible for object recognition. It processes visual information to identify and categorize objects, achieving a balance between selectivity (distinguishing different objects) and invariance (recognizing the same object under various viewing conditions). To give a better idea of the process i will start by describing it from the very first input, the retina, even if the pathway starts from V1.

> The retino-geniculo-striate pathway starts here

1. Retina

   Light arrives already processed by lenses and hits the back of the eye where:

   - Photoreceptors: convert light into neural signals capturing the instensity with rodes and the colour with cones.
   - Bipolar cells: connect the photoreceptors with RGC
   - Retinal Ganglian Cells (RGC): those are neurons with concentric circular receptive fields that can have opposite polarity (on-center and off-center). They are implemented like a difference of two gaussians (narrow center and wide base) to emphasize brightness changes passing by the center. They can detect edges but not their orientation.
2. Optic chiasm: the optic nerves from both the emispheres cross and divide so that each hemisphere receives input from the contralateral visual hemifield.
3. Lateral Geniculate Nucleus (LGN): receives the six layers (three from controlater and three from ipsilateral eye) and store them in a pile implementing a spatial register, splitting the information while maintaining the spatial alignment.

> The ventral visual system starts here.

4. Primary Visual Cortex (V1)

   Recevives the layers from LGN and maintains a precise retinotopic map of the contralateral visual hemifield that is distorted by cortical magnification. The central vision is overrepresented (first 10° occupy 50% of surface).

   It forms a visuotropic map of the vsiual field modelled by the ice cube model that represents a small volume of cortex (a hypercolumn) containing a full set of orientation preferences for both eyes, at a single retinotopic location. Stacking these units across the cortical surface produces a continuous visuotopic map:

   - Orientation columns: neurons that share the same preferred orientation. Adjacent ones shift about 10°, so covering the whole 180° in 12 columns.
   - Ocular Dominance Columns: reflect the LGN segregated inputs, they alternate the inputs from one eye from the other.
   - Blobs: circular regions in superficial layers of the cube that are color-sensitive but orientation-insensitive.
   - Interblobs: regions between blobs that are orientation‑sensitive and color‑insensitive.

   The neurons in this layer can be divided into:

   - Simple cells: stack of receptive fields from LGN cells, forming an ellyptical RF that is sensible to a narrow range of orientations.
   - Complex cells: stack of simple cells' RF, building a grid-like RF that is more sensitive to orientation and movement. They are less sensitive to position and they are detecting edges

   Those cells are hierarchical organized
5. V2: receives strong input from V1, responds to illusory contours, participates in border‑ownership assignment and natural texture processing.
6. V4: used for figure‑ground segregation, curvatures, colors and deals with partially occluded shapes. Receptive fields grow larger.
7. Inferotemporal Cortex (IT)

   It is the highest purely visual stage in the ventral hierarchy. Its job is to create a neural representation that is both selective enough to distinguish between different objects and invariant enough to recognize the same object across changes in viewpoint, size, position, lighting, or partial occlusion.

   There was a debate of how the representation is phisically encoded in the brain:

   - Local/Single encoding: one neuron for one specific concept, as it was observed that some neurons responded to specific peole. This representation is inefficient, fragile and doesn't explain how we generalize so well.
   - Distributed encoding: an object is represented by the unique pattern of activity across a large population of neurons. That explains why we can recognize new objects so well and why can generalize and find similarities between objects

   The representation can be modelled as a response vector, where a population of N neurons maps the representation as a N-dimensional vector. In this way similar objects will be mapped in similar clouds of points and will be easier to seprarate for classification.

## 7.2 What are key similarities and differences between HCNNs and the primate ventral visual pathway?

An HCNN, or Hierarchical Convolutional Neural Network, is a type of artificial neural network designed for processing visual data. It's built in stacked layers, where each layer applies a set of filters to its input, followed by a nonlinear activation and often a pooling operation. The primate visual pathway was described in th [previous answer](#7-1).

They have some similarities and some differences:

- Similarities

  - **Hierarchical Structure**: this is the core similarity. Both systems process visual information in stages.

    - In the brain, this goes from V1 → V2 → V4 → IT cortex.
    - In an HCNN, it's from early convolutional layers → middle layers → deeper layers.

    In both, complexity of the represented features increases with each stage.
  - **Core Computational Operations**: both systems essentially perform a series of Linear-Nonlinear (LN) transformations. A local filter is applied (linear operation), followed by a threshold-like nonlinearity (like ReLU in HCNNs, similar to neural firing thresholds).
  - **Built-in Inductive Biases**: both architectures are biased to process the statistics of the natural world. They assume locality (nearby pixels are related) and shift-invariance (a feature like an edge is important regardless of where it appears in the visual field). In HCNNs, this is baked in through convolutional layers and weight sharing.
  - **Predictive Power**: when optimized for object recognition, HCNNs become surprisingly good functional models of the ventral stream. They can predict neural activity in areas like V4 and IT, and their performance on image categorization tasks correlates with primate behavioral performance.
- Differences

  - **Information flow**:

    - Ventral Pathway: has abundant recurrent (feedback) connections, both within and between areas. This allows for top-down influences (like attention, expectation), contextual integration, and iterative refinement of perceptions over time.
    - HCNN (Standard): are overwhelmingly feedforward. Information flows one way, layer by layer, with no built-in mechanism for feedback or temporal refinement.
  - **Learning Process**:

    - Ventral Pathway: learns unsupervised from a continuous, temporal stream of visual experience. Learning rules are local.
    - HCNN: òearns via supervised training on massive, static, labeled datasets. The learning algorithm (backpropagation) requires a global error signal and weight updates that are biologically implausible.
  - **Robustness & Generalization**:

    - Ventral Pathway: wxtremely robust. Primates excel at recognizing objects from minimal information (heavy occlusion), rely heavily on global shape, and are not fooled by small adversarial pixel changes or texture tricks that confuse AI.
    - HCNN: Can be fragile. They often rely on local texture cues, fail dramatically on heavily occluded objects without special recurrence, and are famously vulnerable to adversarial attacks.
  - **Temporal Dynamics & Challenge Processing**:

    - Ventral Pathway: for easy images, recognition is fast (~100-150ms, feedforward sweep). For challenging images (occluded, noisy), additional processing time (~30-50ms more) is needed, supported by recurrent circuits. This is observable in both slower reaction times and delayed neural solutions in IT cortex.
    - HCNN: has a fixed processing depth. It cannot take "more time" to think about a hard problem. While very deep CNNs may partially approximate this by acting like "unrolled" recurrence, they lack the flexible, dynamic iteration of biological recurrent loops.

## 7.3 Backpropagation and the brain: Similarities and differences.

Backpropagation is a central algorithm for training artificial neural networks (ANNs), whereas learning in the brain is implemented through biologically constrained mechanisms based on synaptic plasticity. Despite major differences in implementation, there are important conceptual similarities.

- **Similarities**

  - **Error-driven learning**:

    Both systems rely on error signals to guide learning. In ANNs, backpropagation minimizes a global loss function by propagating error information backward through the network. In the brain, learning is also influenced by error-related signals, such as prediction errors or reward signals. For example, the cerebellum uses supervised error signals to adjust synaptic strengths, and dopaminergic systems signal discrepancies between expected and obtained outcomes. In both cases, learning is driven by mismatches between predicted and actual states.
  - **Hierarchical and layered processing**:

    Both ANNs and the brain process information through hierarchical stages. DNNs transform inputs across successive layers, with higher layers encoding increasingly abstract representations. Similarly, the cortex is hierarchically organized, as seen in the ventral visual stream, where processing progresses from early sensory areas to higher-level object representations.
  - **Synaptic plasticity as a core mechanism**:

    Learning in both systems depends on changes in connection strength. In ANNs, learning consists of adjusting synaptic weights through gradient-based update rules. In the brain, learning relies on synaptic plasticity mechanisms such as Hebbian learning, long-term potentiation, and activity-dependent changes. In both cases, information is stored primarily by modifying existing connections rather than by creating new neurons.
  - **Use of feedback pathways**:

    Both systems involve feedback connections. Backpropagation requires feedback pathways to transmit error information to earlier layers during learning. In the brain, feedback and recurrent connections are widespread across cortical and thalamocortical circuits. These pathways modulate neural activity, influence plasticity, and are central in theories such as predictive coding, where feedback conveys expectations or error-related signals.
  - **Credit assignment across layers**:

    Both ANNs and the brain must solve the credit assignment problem, namely determining how changes in earlier layers contribute to behavioral outcomes. Backpropagation provides an exact mathematical solution using the chain rule. The brain appears to solve this problem approximately, using local plasticity rules combined with global modulatory signals and network dynamics.
- **Differences**:

  - **Continuous artificial units vs spiking biological neurons**:

    Artificial neurons typically produce continuous-valued outputs, which allow smooth gradients to be computed during learning. Biological neurons, instead, communicate through discrete action potentials (spikes). Information is encoded in spike timing and firing rates rather than continuous activation values. This makes exact gradient computation biologically implausible and requires learning mechanisms based on local activity and temporal correlations rather than explicit derivatives.
  - **Feedforward dominance in ANNs vs pervasive recurrence in the brain**:

    Standard backpropagation is usually implemented in mostly feedforward architectures, where information flows from input to output, and a separate backward pass is used only for learning. In contrast, the brain is highly recurrent. Cortical circuits contain extensive lateral connections, feedback from higher to lower areas, and re-entrant loops. These recurrent interactions contribute not only to learning but also to ongoing perception and cognition, blurring the distinction between forward and backward processing.
  - **Separation of inference and learning vs their entanglement**:

    In ANNs, inference (forward pass) and learning (backward pass) are clearly separated processes. In the brain, perception and learning are deeply intertwined. Neural activity that supports perception also drives synaptic plasticity, and learning can occur during ongoing activity, through delayed feedback, or during offline processes such as replay.
  - **Explicit backward error signals vs implicit error modulation**:

    Backpropagation relies on explicit, signed error signals that are propagated backward through the same layers used for inference. The brain does not transmit explicit error vectors. Instead, error information is conveyed indirectly through neuromodulatory systems (e.g., dopamine), local mismatch signals, and feedback-induced changes in activity, which modulate synaptic plasticity without providing precise gradient information.
  - **Biological constraints on backpropagation**:

    Backpropagation requires symmetric forward and backward weights, access to global error signals, and precise timing of updates. These requirements conflict with biological constraints, since synapses only have access to local signals and biological feedback pathways are coarse, delayed, and multifunctional. As a result, the brain likely implements only approximate, biologically plausible alternatives to backpropagation rather than the algorithm itself.

## 7.4 Deep RL systems are often described as sample-inefficient. What are two potential sources of slowness in deep RL systems, and how do they relate to neuroscience?

Deep RL systems are described as sample-inefficient, meaning that they need more samples than humans to achieve the same level of performance.

This is caused by:

1. **Incremental parameter adjustment**:

   Deep RL typically relies on gradient descent to slowly adjust many network parameters. Updates must be small to avoid instability and catastrophic interference (new learning overwriting old learning). As a result, many experiences are needed before behavior improves reliably.

   This learning style closely resembles how the neocortex works in the brain. In the neocortex, knowledge is stored in widely distributed synaptic connections that are shared across many memories. Because the same synapses support many pieces of information, they can only change gradually. This allows the brain to extract stable, general patterns from experience, but it also makes learning slow when relying on the cortex alone.

   However, biological learning is not limited to this slow mechanism. The brain includes a separate system, the hippocampus, that can rapidly store individual experiences with minimal interference. This fast system allows organisms to learn from single or few events and later replay or integrate those experiences into the neocortex over time.
2. **Weak inductive bias**:

   Many deep RL systems start close to a “tabula rasa”, with few built-in assumptions about task structure. Weak inductive bias allows flexibility across tasks but requires much more data to identify the correct policy. The learner must explore a large hypothesis space, slowing learning.

   Humans and animals come with strong inductive biases shaped by evolution and prior experience (e.g., intuitive physics, task schemas). These biases strongly constrain learning, making it faster and more sample-efficient. In the brain, such biases are supported by multiple memory systems and meta-learning mechanisms that reuse past knowledge.
