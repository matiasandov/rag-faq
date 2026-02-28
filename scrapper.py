import os
import re
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
from dataclasses import dataclass
import json

@dataclass
class DocumentSection:
    """Represents a section of the document"""
    title: str
    content: str
    level: int
    metadata: Dict = None

class PDFScraper:
    """
    General-purpose PDF scraper for extracting structured content
    Designed for Company organizational documents and similar PDFs
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.sections = []
        self.metadata = {}
        
    def extract_text(self) -> str:
        """Extract all text from PDF"""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                # Extract metadata
                if pdf_reader.metadata:
                    self.metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'pages': len(pdf_reader.pages)
                    }
                
                # Extract text from all pages
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                
                self.raw_text = '\n'.join(text_content)
                return self.raw_text
                
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_acronyms(self) -> Dict[str, str]:
        """Extract acronym definitions from document"""
        acronyms = {}
        
        # Pattern to match acronym tables
        acronym_pattern = r'([A-Z&]+)\s+([A-Za-z\s&]+?)(?=\n[A-Z&]+\s|\n\n|\Z)'
        
        # Find acronym section
        acronym_section = re.search(
            r'List of Acronyms.*?(?=\n[A-Z][a-z]+|\nContents|\Z)', 
            self.raw_text, 
            re.DOTALL | re.IGNORECASE
        )
        
        if acronym_section:
            matches = re.findall(acronym_pattern, acronym_section.group())
            for acronym, meaning in matches:
                acronyms[acronym.strip()] = meaning.strip()
        
        return acronyms
    
    def identify_sections(self) -> List[DocumentSection]:
        """Identify and extract document sections"""
        sections = []
        
        # Split by common section patterns
        lines = self.raw_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers (various patterns)
            if self._is_section_header(line):
                # Save previous section
                if current_section:
                    sections.append(DocumentSection(
                        title=current_section,
                        content='\n'.join(current_content).strip(),
                        level=self._get_header_level(current_section)
                    ))
                
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections.append(DocumentSection(
                title=current_section,
                content='\n'.join(current_content).strip(),
                level=self._get_header_level(current_section)
            ))
        
        self.sections = sections
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is a section header"""
        header_patterns = [
            r'^[A-Z][A-Za-z\s&]+:$',  # Ends with colon
            r'^•\s+[A-Z]',  # Bullet point with capital
            r'^o\s+[A-Z]',  # 'o' bullet with capital
            r'^Phase\s+\d+:',  # Phase headers
            r'^Process\s+\w+:',  # Process headers
            r'^(Main Activities|Skill Team|Process Name)',  # Key sections
        ]
        
        return any(re.match(pattern, line) for pattern in header_patterns)
    
    def _get_header_level(self, line: str) -> int:
        """Determine the hierarchical level of a header"""
        if line.startswith('•'):
            return 1
        elif line.startswith('o'):
            return 2
        elif 'Phase' in line:
            return 2
        else:
            return 0
    
    def extract_processes(self) -> List[Dict]:
        """Extract process information from document"""
        processes = []
        
        # Find process sections
        process_pattern = r'Process Name:\s*(.*?)(?=Process Name:|$)'
        matches = re.finditer(process_pattern, self.raw_text, re.DOTALL)
        
        for match in matches:
            process_text = match.group(1)
            
            process_info = {
                'name': self._extract_field(process_text, r'Process Name:\s*(.+)'),
                'description': self._extract_field(process_text, r'Process Description:\s*(.+?)(?=\n•|\nProcess Owner:)'),
                'owner': self._extract_field(process_text, r'Process Owner:\s*(.+)'),
                'areas_involved': self._extract_field(process_text, r'Areas Involved:\s*(.+)'),
                'manager': self._extract_field(process_text, r'Area Manager.*?:\s*(.+)'),
                'status': self._extract_field(process_text, r'Process Status:\s*(.+)'),
                'phases': self._extract_phases(process_text)
            }
            
            processes.append(process_info)
        
        return processes
    
    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        """Extract a specific field using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_phases(self, text: str) -> List[Dict]:
        """Extract process phases"""
        phases = []
        phase_pattern = r'Phase\s+(\d+):\s*(.+?)(?=Phase\s+\d+:|Process Status:|$)'
        
        matches = re.finditer(phase_pattern, text, re.DOTALL)
        for match in matches:
            phase_num = match.group(1)
            phase_content = match.group(2).strip()
            
            # Extract phase title and steps
            lines = phase_content.split('\n')
            title = lines[0].strip() if lines else ""
            steps = [line.strip('• ').strip() for line in lines[1:] if line.strip()]
            
            phases.append({
                'number': int(phase_num),
                'title': title,
                'steps': steps
            })
        
        return phases
    
    def chunk_for_rag(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Create chunks suitable for RAG applications
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Character overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        #for each section we extract text
        for section in self.sections:
            text = f"{section.title}\n{section.content}"
            
            # Split into chunks with overlap
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                chunks.append({
                    'id': f"{Path(self.pdf_path).stem}_chunk_{len(chunks)}",
                    'text': chunk_text,
                    'metadata': {
                        'source': self.pdf_path,
                        'section': section.title,
                        'section_level': section.level,
                        'chunk_index': chunk_id,
                        **self.metadata
                    }
                })
                
                start = end - overlap
                chunk_id += 1
        
        return chunks
    
    def to_json(self, output_path: Optional[str] = None) -> str:
        """Export extracted data to JSON"""
        data = {
            'metadata': self.metadata,
            'acronyms': self.extract_acronyms(),
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level
                } for s in self.sections
            ],
            'processes': self.extract_processes(),
            'chunks': self.chunk_for_rag()
        }
        
        json_output = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
        
        return json_output
    

# # Example usage
# if __name__ == "__main__":
#     # Process single PDF
#     scraper = PDFScraper("accesorios.pdf")
#     data = scraper.process()
    
#     # Export to JSON
#     scraper.to_json("accesorios_extracted.json")
    
#     # Print summary
#     print(f"Extracted {len(data['sections'])} sections")
#     print(f"Found {len(data['acronyms'])} acronyms")
#     print(f"Identified {len(data['processes'])} processes")
#     print(f"Created {len(data['chunks'])} chunks for RAG")
    
#     # Get chunks for RAG
#     chunks = scraper.chunk_for_rag(chunk_size=500, overlap=50)
#     for chunk in chunks[:3]:  # Show first 3 chunks
#         print(f"\nChunk ID: {chunk['id']}")
#         print(f"Text preview: {chunk['text'][:100]}...")
