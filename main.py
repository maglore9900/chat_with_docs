import adaptor
import argparse
import sys





def main():
    parser = argparse.ArgumentParser(description="Document loader and query processor")
    parser.add_argument("-d", "--document", required=False, help="Path to the document to load")
    parser.add_argument("-q", "--query", required=False, help="Query to search in the document or vector db")
    parser.add_argument("-v", "--vector", action="store_true" , required=False, help="Select if you want to query the vectordb.\nYou must create a vectordb first, by adding one or more files to it")
    parser.add_argument("-a", "--add_to_vectordb", action="store_true", required=False, help="Adds the file into a vector db")
    args = parser.parse_args()
    
    # Load the document
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ad = adaptor.Adaptor()
    if (args.query and args.document) or (args.query and args.vector) or (args.document and args.add_to_vectordb):
        if args.add_to_vectordb and args.document:
            ad.add_to_datastore(args.document)
        elif args.query and args.document:
            # doc = ad.vector_doc(args.document)
            result = ad.query_doc(args.query, args.document)
            print(result)
        elif args.query and args.vector:
            result = ad.query_datastore(args.query)
            print(result)
    else:
        parser.print_help()
        print("Please make a valid selection.\nQuery and a source (document or vectordb), or add to datastore and document")


if __name__ == "__main__":
    main() 