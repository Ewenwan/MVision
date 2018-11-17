%function fname = genfilename(sequencestruct, framenumber)
function fname = genfilename(sequencestruct, framenumber)
digstr = sprintf('%%0%dd',sequencestruct.digits);
filstr = sprintf('%%s%s%%s',digstr);
fname = sprintf(filstr,sequencestruct.prefix,framenumber,sequencestruct.postfix);
return

