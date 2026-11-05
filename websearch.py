# import os
# import json
# import re
# from typing import Optional, List
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_core.tools import tool
# from langchain.agents import create_agent
# from dotenv import load_dotenv
# from exa_py import Exa
# import requests
# from bs4 import BeautifulSoup

# load_dotenv()

# # ============================================================================
# # STEP 1: DEFINE THE DATA STRUCTURE YOU WANT
# # ============================================================================

# class SupplierProduct(BaseModel):
#     """Structured product information from a supplier"""
#     company_name: str = Field(description="Name of the supplier company")
#     website: str = Field(description="Company website URL")
#     product_name: str = Field(description="Exact product name/SKU")
#     price_per_unit: Optional[float] = Field(None, description="Price per unit in USD")
#     currency: str = Field(default="USD", description="Currency code")
#     minimum_order_quantity: Optional[int] = Field(None, description="MOQ if specified")
#     rating: Optional[float] = Field(None, description="Product/supplier rating out of 5")
#     review_count: Optional[int] = Field(None, description="Number of reviews")
#     delivery_time_days: Optional[int] = Field(None, description="Expected delivery in days")
#     in_stock: Optional[bool] = Field(None, description="Availability status")
#     certifications: List[str] = Field(default_factory=list, description="ISO, CE, etc.")

# class SupplierList(BaseModel):
#     """Collection of suppliers for a specific product"""
#     search_query: str
#     suppliers: List[SupplierProduct]


# # ============================================================================
# # STEP 2: CREATE THE SEARCH TOOL
# # ============================================================================
# EXA_API_KEY=os.getenv("EXA_API_KEY")
# if not EXA_API_KEY:
#     raise RuntimeError(
#         "EXA_API_KEY is missing. Add EXA_API_KEY=your_key to .env or export it in the shell."
#     )
# exa_client = Exa(api_key=EXA_API_KEY)

# @tool
# def discover_suppliers(product: str, region: str = "global", max_results: int = 10) -> str:
#     """
#     Find B2B suppliers for a product using neural search.
#     Returns company websites and descriptions.
#     """
#     query = f"{product} wholesale supplier manufacturer B2B"
#     if region != "global":
#         query += f" in {region}"
        
#     results = exa_client.search(
#         query,
#         type="neural",
#         category="company",
#         num_results=max_results,
#         contents={"text": {"max_characters": 1000}}
#     )
    
#     suppliers = []
#     for r in results.results:
#         suppliers.append({
#             "company_name": r.title,
#             "website": r.url,
#             "description": r.text
#         })
        
#     return json.dumps(suppliers, indent=2)


# # ============================================================================
# # STEP 3: CREATE THE SCRAPING + EXTRACTION TOOL
# # ============================================================================

# @tool
# def extract_product_details(url: str, product_name: str) -> str:
#     """
#     Scrape a supplier's website and extract structured product information:
#     price, rating, delivery time, and availability.
    
#     Args:
#         url: The supplier's product page or homepage
#         product_name: The product you're looking for (helps focus extraction)
#     """
#     try:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (procurement-bot/1.0)"
#         }
#         response = requests.get(url, timeout=15, headers=headers)
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.text, 'html.parser')
#         page_text = soup.get_text(separator=" ", strip=True)
        
#         # Use LLM to extract structured data from the scraped HTML
#         # (We'll do this in a sub-call below)
        
#         # For now, return the raw text for the LLM to parse
#         return page_text[:3000]  # Limit to 3000 chars to fit in context
        
#     except Exception as e:
#         return json.dumps({"error": f"Could not scrape {url}: {str(e)}"})


# # ============================================================================
# # STEP 4: CREATE AN EXTRACTION-FOCUSED LLM
# # ============================================================================

# extraction_llm = ChatGroq(
#     model="qwen/qwen3.8-27b",
#     temperature=0
# ).with_structured_output(SupplierProduct)

# def extract_structured_data(url: str, raw_html_text: str, product_query: str) -> SupplierProduct:
#     """
#     Use Qwen to extract structured product data from scraped webpage text.
#     """
#     extraction_prompt = f"""You are a data extraction specialist for procurement.

# Extract the following information from this supplier webpage about "{product_query}":

# REQUIRED:
# - Company name
# - Product name/SKU
# - Price per unit (convert to USD if needed)
# - Website URL

# OPTIONAL (extract if found, otherwise null):
# - Rating (out of 5 stars)
# - Number of reviews
# - Delivery/lead time in days
# - Minimum order quantity (MOQ)
# - Stock status (in stock / out of stock / made to order)
# - Certifications (ISO 9001, CE, FDA, etc.)

# RULES:
# - Only extract information explicitly stated on the page
# - If price is a range, use the lower bound
# - If delivery says "2-3 weeks", convert to days (14-21 → use 21)
# - Do NOT invent data. Use null for missing fields.

# WEBPAGE TEXT:
# {raw_html_text}

# URL: {url}
# """
    
#     result = extraction_llm.invoke(extraction_prompt)
#     result.website = url  # Ensure the URL is set
#     return result


# # ============================================================================
# # STEP 5: COMBINE INTO A UNIFIED WORKFLOW TOOL
# # ============================================================================

# @tool
# def find_suppliers_with_pricing(product: str, region: str = "india") -> str:
#     """
#     Complete supplier discovery tool that returns:
#     - Company name
#     - Product price
#     - Rating
#     - Delivery time
    
#     This is a high-level tool that internally calls search + scraping + extraction.
#     """
#     # Step 1: Search for suppliers
#     search_results = json.loads(discover_suppliers.invoke({"product": product, "region": region}))
    
#     # Step 2: For each supplier, scrape and extract details
#     detailed_suppliers = []
    
#     for supplier in search_results[:5]:  # Limit to top 5 to avoid long runtime
#         try:
#             # Scrape the page
#             raw_text = extract_product_details.invoke({
#                 "url": supplier["website"],
#                 "product_name": product
#             })
            
#             # Extract structured data
#             structured = extract_structured_data(
#                 url=supplier["website"],
#                 raw_html_text=raw_text,
#                 product_query=product
#             )
            
#             detailed_suppliers.append(structured.model_dump())
            
#         except Exception as e:
#             # If scraping fails, still include basic info
#             detailed_suppliers.append({
#                 "company_name": supplier["company_name"],
#                 "website": supplier["website"],
#                 "error": str(e)
#             })
    
#     return json.dumps(detailed_suppliers, indent=2)


# # ============================================================================
# # STEP 6: BUILD THE BUYER AGENT
# # ============================================================================

# llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

# tools = [find_suppliers_with_pricing]  # The unified tool

# system_prompt = """You are a procurement Buyer Agent.

# Your job: Find suppliers with complete pricing and delivery information.

# PROCESS:
# 1. Use `find_suppliers_with_pricing` to get detailed supplier data
# 2. Filter suppliers that have:
#    - Valid price information
#    - Acceptable delivery time (< 30 days preferred)
#    - Good rating (> 4.0 stars preferred)
# 3. Rank suppliers by: price (lower better) > delivery time (faster better) > rating (higher better)
# 4. Return the top 3 suppliers with all details

# OUTPUT FORMAT:
# For each supplier, provide:
# - Company name
# - Price per unit
# - Rating
# - Delivery time
# - Why you ranked them in this position
# """

# buyer_agent = create_agent(
#     model=llm,
#     tools=tools,
#     system_prompt=system_prompt,
# )


# # ============================================================================
# # STEP 7: RUN IT
# # ============================================================================

# if __name__ == "__main__":
#     result = buyer_agent.invoke({
#         "messages": [{
#             "role": "user",
#             "content": "Find suppliers for 1 TB NVMe M.2 SSD. I need price, ratings, and delivery time."
#         }]
#     })
    
#     print("\n" + "="*80)
#     print("FINAL RESULT:")
#     print("="*80)
#     print(result["messages"][-1].content)

import os
import json
import re
import uuid
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from serpapi import GoogleSearch
from buyer_negotiation import DEFAULT_NEGOTIATION_LOG, log_token_usage

# Load environment variables
load_dotenv()

# ============================================================================
# DATA MODELS
# ============================================================================

class SupplierProduct(BaseModel):
    """Structured product information from suppliers"""
    rank: int = Field(description="Ranking position")
    company_name: str = Field(description="Seller/retailer name")
    product_name: str = Field(description="Full product name")
    price: str = Field(description="Price as displayed (e.g., '$89.99')")
    price_numeric: Optional[float] = Field(None, description="Price as number for comparison")
    currency: str = Field(default="USD", description="Currency code")
    rating: Optional[float] = Field(None, description="Product rating out of 5")
    review_count: Optional[int] = Field(None, description="Number of reviews")
    url: str = Field(description="Product page URL")
    delivery: Optional[str] = Field(None, description="Delivery estimate")
    in_stock: Optional[bool] = Field(None, description="Stock availability")

class SupplierList(BaseModel):
    """Collection of suppliers for comparison"""
    search_query: str
    total_found: int
    suppliers: List[SupplierProduct]

# ============================================================================
# GOOGLE SHOPPING SEARCH TOOL
# ============================================================================

@tool
def search_google_shopping(product: str, max_results: int = 10) -> str:
    """
    Search Google Shopping for products with pricing, ratings, and seller information.
    Returns structured data ready for procurement evaluation.
    
    Args:
        product: Product name to search for (e.g., "1TB NVMe M.2 SSD")
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        JSON string with supplier details including price, rating, delivery
    """
    print(f"🔍 Searching Google Shopping for: {product}")
    
    params = {
        "engine": "google_shopping",
        "q": product,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": max_results,
        "gl": "us",  # Country code (us, in, uk, etc.)
        "hl": "en"   # Language
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        suppliers = []
        
        if "shopping_results" in results:
            print(f"   ✅ Found {len(results['shopping_results'])} products")
            
            for idx, item in enumerate(results["shopping_results"][:max_results]):
                # Extract and parse price
                price_str = item.get("price", "")
                price_numeric = None
                
                if price_str:
                    # Handle formats like "$89.99", "€45.50", "£99"
                    price_match = re.search(r'[\$€£¥₹]?([\d,]+\.?\d*)', price_str)
                    if price_match:
                        price_numeric = float(price_match.group(1).replace(',', ''))
                
                # Extract rating (can be float or string like "4.5")
                rating = item.get("rating")
                if rating and isinstance(rating, str):
                    rating_match = re.search(r'([\d.]+)', rating)
                    rating = float(rating_match.group(1)) if rating_match else None
                
                # Extract review count
                reviews = item.get("reviews")
                review_count = None
                if reviews:
                    if isinstance(reviews, int):
                        review_count = reviews
                    else:
                        # Handle formats like "1,234 reviews" or "(523)"
                        review_match = re.search(r'([\d,]+)', str(reviews))
                        if review_match:
                            review_count = int(review_match.group(1).replace(',', ''))
                
                # Check stock status
                in_stock = None
                if "tag" in item:
                    tag = item["tag"].lower()
                    if "in stock" in tag:
                        in_stock = True
                    elif "out of stock" in tag:
                        in_stock = False
                
                # Build supplier record
                supplier = {
                    "rank": idx + 1,
                    "company_name": item.get("source", "Unknown Seller"),
                    "product_name": item.get("title", "N/A"),
                    "price": price_str if price_str else "Contact for quote",
                    "price_numeric": price_numeric,
                    "currency": "USD",  # Can be extended to detect currency from price_str
                    "rating": rating,
                    "review_count": review_count,
                    "url": item.get("link", ""),
                    "delivery": item.get("delivery", None),
                    "in_stock": in_stock
                }
                
                suppliers.append(supplier)
            
            # Sort by price (cheapest first)
            suppliers.sort(key=lambda x: x['price_numeric'] if x['price_numeric'] else float('inf'))
            
            # Re-rank after sorting
            for i, s in enumerate(suppliers):
                s['rank'] = i + 1
            
            print(f"   📊 Returning {len(suppliers)} sorted results")
            return json.dumps(suppliers, indent=2)
        
        else:
            print("   ⚠️ No shopping results found")
            return json.dumps([], indent=2)
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        print(f"   ❌ {error_msg}")
        return json.dumps({"error": error_msg}, indent=2)


# ============================================================================
# BACKUP TOOL: Search for B2B/Wholesale Pricing
# ============================================================================

@tool
def search_wholesale_suppliers(product: str, max_results: int = 10) -> str:
    """
    Search specifically for wholesale/bulk suppliers.
    Adds 'wholesale' and 'bulk' keywords to find B2B pricing.
    """
    print(f"🏭 Searching for wholesale suppliers: {product}")
    
    # Modify query to focus on B2B
    wholesale_query = f"{product} wholesale bulk supplier"
    
    params = {
        "engine": "google_shopping",
        "q": wholesale_query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": max_results,
        "gl": "us",
        "hl": "en"
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        suppliers = []
        
        if "shopping_results" in results:
            for idx, item in enumerate(results["shopping_results"][:max_results]):
                price_str = item.get("price", "")
                price_numeric = None
                
                if price_str:
                    price_match = re.search(r'[\$€£]?([\d,]+\.?\d*)', price_str)
                    if price_match:
                        price_numeric = float(price_match.group(1).replace(',', ''))
                
                suppliers.append({
                    "rank": idx + 1,
                    "company_name": item.get("source", "Unknown"),
                    "product_name": item.get("title", "N/A"),
                    "price": price_str if price_str else "RFQ Required",
                    "price_numeric": price_numeric,
                    "url": item.get("link", ""),
                    "note": "Wholesale/Bulk listing"
                })
            
            suppliers.sort(key=lambda x: x['price_numeric'] if x['price_numeric'] else float('inf'))
            
            for i, s in enumerate(suppliers):
                s['rank'] = i + 1
            
            return json.dumps(suppliers, indent=2)
        
        return json.dumps([], indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ============================================================================
# BUYER AGENT SETUP
# ============================================================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0  # Deterministic for procurement decisions
)

tools = [search_google_shopping, search_wholesale_suppliers]

system_prompt = """You are an autonomous procurement Buyer Agent.

Your job: Find suppliers with complete pricing and rating information for the requested product.

SEARCH STRATEGY:
1. FIRST: Use `search_google_shopping` - gets retail pricing with ratings
2. IF user needs bulk/wholesale: Use `search_wholesale_suppliers` 
3. Compare and rank results

EVALUATION CRITERIA:
- **Price**: Lower is better (but flag suspiciously cheap items)
- **Rating**: Prefer 4.5+ stars with 100+ reviews
- **Availability**: In-stock products rank higher
- **Delivery**: Faster delivery preferred
- **Seller Reputation**: Known retailers (Amazon, Newegg, B&H) are safer

OUTPUT FORMAT:
1. **Summary Table** (Top 5-10 suppliers in markdown)
2. **Recommendation** with clear reasoning
3. **Concerns** - Flag any issues (low rating, unknown seller, out of stock)

RULES:
- Never invent prices or ratings
- If no pricing data: say "Contact for quote"
- Show price per unit clearly
- For bulk orders, mention to contact seller directly for volume discounts
"""

buyer_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def display_results(result: dict):
    """Pretty print the agent's output"""
    print("\n" + "="*80)
    print("📦 PROCUREMENT SEARCH RESULTS")
    print("="*80)
    print(result["messages"][-1].content)
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🤖 Buyer Agent Started")
    print("="*80 + "\n")
    
    # Example query - you can modify this
    query = "Find suppliers for 8mm Hex Bolt with pricing and ratings"
    
    print(f"📝 Query: {query}\n")
    
    search_run_id = f"supplier-search-{uuid.uuid4()}"
    result = buyer_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    llm_call_number = 0
    for message in result["messages"]:
        if getattr(message, "type", None) != "ai":
            continue
        llm_call_number += 1
        response_metadata = getattr(message, "response_metadata", {})
        log_token_usage(
            DEFAULT_NEGOTIATION_LOG,
            agent="buyer_agent",
            model=(
                response_metadata.get("model_name")
                or getattr(llm, "model", None)
                or getattr(llm, "model_name", None)
            ),
            usage_metadata=getattr(message, "usage_metadata", None),
            session_id=f"{search_run_id}-call-{llm_call_number}",
        )

    display_results(result)
    
    # Save results to file
    with open("procurement_results.json", "w") as f:
        json.dump({
            "query": query,
            "output": result["messages"][-1].content,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n💾 Results saved to procurement_results.json")