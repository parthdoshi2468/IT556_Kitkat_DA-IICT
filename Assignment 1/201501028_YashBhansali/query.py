#connect to our cluster
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
print(es.get(index='music', doc_type='data', id=5))
print es.search(index="music",body=
{
  "query": { "match": { "genre": "pop"} },
  })

print es.search(index="music",body=
{
  "query": { "match": { "artist": "Jones Watkins"} },
  })
  
print es.search(index="music",body=
{
  "query": { 
    "bool": { 
      "must": [
        { "match": { "genre":   "rock"}}
      ],
      "filter": [ 
        { "range": { "length": { "lte": 5 }}} ,
        { "range": { "length": { "gte": 3 }}} 
      ]
    }
  }
})