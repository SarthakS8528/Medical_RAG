from query_expansion import expand_query
queries=[
      "What causes insulin resistance in type 2 diabetes?",
      "How does the meal timing affect blood  glucose control in type 2 diabetes?",
      "What is the role of ferroptosis in pancreatic beta cell dysfunction?"
        ]   

for query in queries:
    expanded=expand_query(query)
    print("=" * 80)
    print("original:")
    print(query)
    print("Expanded:")
    print(expanded)     