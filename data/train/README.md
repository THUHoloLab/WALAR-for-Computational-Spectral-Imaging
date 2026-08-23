# Training Data

Place `data.mdb` and `lock.mdb` in this directory. The LMDB stores indexed
measurement--target pairs and their normalized target wavelengths. Using one
database limits small-file I/O overhead for the large training split.
