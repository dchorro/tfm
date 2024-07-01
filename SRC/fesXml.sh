N_COLS=`file $1|awk -F"," '{  split($(NF-1),s,"x"); print s[1]}'`
N_ROWS=`file $1|awk -F"," '{  split($(NF-1),s,"x"); print s[2]}'`

nom=`basename $1`
awk -v n_rows=`echo $N_ROWS` -v n_cols=`echo $N_COLS` -v fileName=$nom '{
      if ($0 ~ /imageFilename/)
          print "        <Page imageFilename=\"" fileName "\" imageWidth=\"" n_cols "\"  imageHeight=\"" n_rows"\">"
      else 
          print

}' Page.xml

