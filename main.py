import adaptor
import argparse

ad = adaptor.Adaptor()



def main():
    parser = argparse.ArgumentParser(description="Document loader and query processor")
    
    parser.add_argument("-d", "--document", required=False, help="Path to the document to load")
    parser.add_argument("-q", "--query", required=False, help="Query to search in the document or vector db")
    parser.add_argument("-v", "--vector-db", required=False, help="Path to the vector database, easiest if you just give it a name in the same dir such as 'vector_db'")
    
    args = parser.parse_args()
    
    # Load the document
    
    if args.document and args.query or args.document and args.vector_db:
        if args.vector_db:
            ad.add_to_datastore(args.document, args.vector_db)
        elif args.query:
            doc = ad.vector_doc(args.document)
            result = ad.query_doc(args.query, doc)
            print(result)
    if args.vector_db and args.query:
        result = ad.query_datastore(args.query, args.vector_db)
        print(result)


if __name__ == "__main__":
    main() 