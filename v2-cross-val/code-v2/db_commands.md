# Frequently Used DB Commands

Saving intermediate data in tables/collections
* so that they don't need to be re-created every time something down the pipeline goes wrong,
* can use the same samples/splits/folds to compare different features/base classifiers

<br>
MongoDB:
```
db.getCollection('sample-tokens').renameCollection('sample-tokens-satoshi-satoshi')


db.getCollectionNames().forEach( function (d) {
  if (d != "web-scrape" && d != "raw-docs") {  
        print("dropping: " + d);
        db[d].drop();
  }
})
```

Postgres:
```sql
ALTER TABLE stats RENAME TO stats_satoshi_satoshi;
ALTER INDEX stats_pkey RENAME TO stats_satoshi_satoshi_pkey;

CREATE TABLE results_backup (
   like results
   INCLUDING DEFAULTS  
   INCLUDING INDEXES
   INCLUDING STORAGE
);
INSERT INTO results_backup
(SELECT * FROM results);


DELETE FROM results_backup;
INSERT INTO results_backup
(SELECT * FROM results);
```
