# Plan

This codebase provides a way to visualize bio2token ([code](https://github.com/flagshippioneering/bio2token) [paper](https://arxiv.org/pdf/2410.19110)) encodings in a browser.

On the left pane of the webpage, we have a bio2token-encoded document, for example:

```
<b2239> <b2751> <b2619> <b1082>
<b3131> <b3127> <b1591> <b2847>
<b1343> <b559> <b1831> <b802>
<b3899> <b3879> <b1819> <b1794>
<b1855> <b1871> <b551> <b514>
<b1851> <b43> <b563> <b514>
<b1847> <b2866> <b1811> <b1857>
<b807> <b2823> <b519> <b642>
<b551> <b799> <b579> <b1809>
<b775> <b834> <b274> <b1>
<b531> <b23> <b529> <b2849>
<b514> <b513> <b561> <b2080>
<b803> <b33> <b1847> <b1083>
<b787> <b2879> <b3890> <b3767>
<b2834> <b3923> <b2865> <b4081>
<b1826> <b2592> <b2099> <b299>
<b3635> <b3387> <b54> <b32>
<b567> <b283> <b1081> <b1052>
<b1082> <b2110> <b2097> <b3096>
<b1330> <b3377> <b304> <b2048>
<b307> <b530> <b37> <b321>
<b55> <b45> <b2084> <b1092>
<b1073> <b3108> <b1312> <b2176>
<b305> <b2848> <b273> <b1152>
<b33> <b7> <b1048> <b1156>
<b1076> <b1068> <b2064> <b2180>
<b1056> <b3104> <b1280> <b1216>
<b288> <b768> <b4> <b1216>
<b20> <b9> <b2052> <b1220>
<b1040> <b3080> <b1344> <b2240>
<b256> <b3344> <b64> <b2752>
<b1040> <b321> <b1160> <b200>
<b2052> <b1100> <b2244> <b1220>
```

The encoding above is one example. It's fine if it's actually a simpler representation such as "2239 2751 2619" and so on.

On the right pane in the browser window we have a live rendering of the 3D structure decoded from the tokenized input on the left pane.

There should also be a mechanism for a user to provide a link to a cif or pdb file. The contents should be tokenized and put in the left pane. On the right pane that shows the decoded structure we should also show the original structure so deviations are obvious. 

It would be great if this can run on github with no backend server. But I imagine there's no way to run the bio2token encoder/decoder without having a server running? In that case I want a simple command (using uv) that will launch the server on my node and tell me what address to go to.

